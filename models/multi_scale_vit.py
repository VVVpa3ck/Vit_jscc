import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

__all__ = [
    "SemanticImportance",
    "FineGrainedExtractor",
    "CoarseGrainedExtractor",
    "CrossAttentionFusion",
    "MultiScaleEncoder",
    "DeepSCRI",
]


class SemanticImportance(nn.Module):
    """Estimates semantic importance for each token and returns a binary mask
    indicating the *least* important tokens to suppress (set attention to -Inf).
    """

    def __init__(self, emb_dim: int, drop_ratio: float = 0.25):
        super().__init__()
        self.score_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, 1),
        )
        self.drop_ratio = drop_ratio  # proportion of tokens to mask per sample

    def forward(self, x: torch.Tensor):
        """x: (B, N, C) tokens incl. CLS; returns (B, N) mask with 0 for keep, -Inf for drop."""
        scores = self.score_head(x).squeeze(-1)  # (B, N)
        k = (scores.size(1) * self.drop_ratio).__round__()
        topk = scores.topk(k, largest=False, dim=1).indices  # least important indices
        mask = torch.zeros_like(scores).fill_(0)
        mask.scatter_(1, topk, float('-inf'))
        return mask.unsqueeze(-1)  # (B, N, 1)


class _PatchEmbedding(nn.Module):
    def __init__(self, in_channels, emb_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, emb_dim, patch_size, patch_size)
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        return self.norm(x)


class FineGrainedExtractor(nn.Module):
    def __init__(self, in_channels, img_size, patch_size, emb_dim, num_layers, num_heads, drop_ratio=0.25):
        super().__init__()
        self.patch = _PatchEmbedding(in_channels, emb_dim, patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.pos = nn.Parameter(torch.zeros(1, num_patches, emb_dim))
        self.blocks = nn.Sequential(*[Block(emb_dim, num_heads) for _ in range(num_layers)])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.semantic_importance = SemanticImportance(emb_dim, drop_ratio)
        self._init_weight()

    def _init_weight(self):
        trunc_normal_(self.pos, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.size(0)
        x = self.patch(x)  # (B, N, C)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x + self.pos], dim=1)
        # compute mask and apply to attention scores via hook
        mask = self.semantic_importance(x)  # (B, N+1, 1)

        def attn_pre_hook(module, input):
            qk, v = input
            qk = qk + mask.transpose(1, 2)  # broadcast mask to (B, 1, N)
            return (qk, v)

        # register temporary hook on all attention blocks
        hooks = []
        for blk in self.blocks:
            hooks.append(blk.attn.qkv.register_forward_pre_hook(attn_pre_hook))
        x = self.blocks(x)
        for h in hooks:
            h.remove()
        return x  # (B, N+1, C)


class CoarseGrainedExtractor(nn.Module):
    def __init__(self, in_channels, img_size, patch_size, emb_dim, num_layers, num_heads):
        super().__init__()
        self.patch = _PatchEmbedding(in_channels, emb_dim, patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.pos = nn.Parameter(torch.zeros(1, num_patches, emb_dim))
        self.blocks = nn.Sequential(*[Block(emb_dim, num_heads) for _ in range(num_layers)])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        # hierarchical pooling: 3-level avg pooling over tokens (except CLS)
        self.levels = 3
        self._init_weight()

    def _init_weight(self):
        trunc_normal_(self.pos, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.size(0)
        x = self.patch(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x + self.pos], dim=1)  # (B, N+1, C)
        x = self.blocks(x)

        # hierarchical pooling on patch tokens
        tokens = x[:, 1:]  # drop cls
        n = tokens.size(1)
        levels = [tokens]
        for _ in range(self.levels - 1):
            n = int(n ** 0.5)
            if n < 1:
                break
            tokens = rearrange(tokens, "b (h w) c -> b h w c", h=int(tokens.size(1) ** 0.5))
            tokens = torch.nn.functional.avg_pool2d(tokens.permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1)
            tokens = rearrange(tokens, "b h w c -> b (h w) c")
            levels.append(tokens)
        multi_scale = torch.cat(levels, dim=1)  # concat along token dim
        return torch.cat([x[:, :1], multi_scale], dim=1)  # prepend CLS


class CrossAttentionFusion(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        self.num_heads = num_heads
        self.scale = (emb_dim // num_heads) ** -0.5
        self.out = nn.Linear(emb_dim, emb_dim)

    def forward(self, fine, coarse):
        """fine, coarse: (B, N, C) with CLS already removed or kept as needed."""
        fused = torch.cat([fine, coarse], dim=1)
        Q = self.q_proj(fine)  # only queries from fine branch
        K = self.k_proj(fused)
        V = self.v_proj(fused)
        B, Nq, C = Q.shape
        Nh = self.num_heads
        Q = Q.view(B, Nh, Nq, C // Nh)
        K = K.view(B, Nh, -1, C // Nh)
        V = V.view(B, Nh, -1, C // Nh)
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ V).reshape(B, Nq, C)
        return self.out(x)


class MultiScaleEncoder(nn.Module):
    def __init__(self, in_channels=3, img_size=224):
        super().__init__()
        # hyper‑params could be exposed
        self.fine = FineGrainedExtractor(in_channels, img_size, patch_size=2, emb_dim=192, num_layers=4, num_heads=3)
        self.coarse = CoarseGrainedExtractor(in_channels, img_size, patch_size=4, emb_dim=192, num_layers=4, num_heads=3)
        self.fusion = CrossAttentionFusion(emb_dim=192, num_heads=3)

    def forward(self, x):
        f = self.fine(x)
        c = self.coarse(x)
        fused = self.fusion(f, c)
        # prepend cls from fine branch (first token)
        cls = f[:, :1]
        tokens = torch.cat([cls, fused], dim=1)
        return tokens.transpose(0, 1)  # (T, B, C) to match existing decoder convention


class Identity(nn.Module):
    def forward(self, x):
        return x


class DeepSCRI(nn.Module):
    """Simplified implementation merging multi‑scale encoder with ViT decoder and optional channel codec."""

    def __init__(self, in_channels=3, img_size=224, emb_dim=192, channel_encoder: nn.Module | None = None, channel_decoder: nn.Module | None = None):
        super().__init__()
        self.semantic_encoder = MultiScaleEncoder(in_channels, img_size)
        self.channel_encoder = channel_encoder or Identity()
        self.channel_decoder = channel_decoder or Identity()
        from models.decoder import ViTDecoder  # reuse existing
        self.semantic_decoder = ViTDecoder(in_channels, img_size, patch_size=4, emb_dim=emb_dim, num_layer=4, num_head=3)

    def forward(self, img):
        semantics = self.semantic_encoder(img)  # (T, B, C)
        tx = self.channel_encoder(semantics)
        rx = self.channel_decoder(tx)
        out = self.semantic_decoder(rx)
        return out
