import torch
import torch.nn.functional as F


def fgsm_attack(model, images, epsilon):
    """
    FGSM for image reconstruction models (e.g., DAEViT).

    Args:
        model: reconstruction model
        images: (B, C, H, W) input
        epsilon: perturbation strength

    Returns:
        perturbed images
    """
    images = images.clone().detach().to(images.device)
    images.requires_grad = True

    outputs = model(images)
    loss = F.mse_loss(outputs, images)
    model.zero_grad()
    loss.backward()

    perturbed = images + epsilon * images.grad.sign()
    perturbed = torch.clamp(perturbed, 0, 1)
    return perturbed


def add_noise_with_snr(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    signal_power = x.pow(2).mean()
    snr = 10 ** (snr_db / 10)
    noise_power = signal_power / snr
    noise = torch.randn_like(x) * torch.sqrt(noise_power)
    return torch.clamp(x + noise, 0, 1)
