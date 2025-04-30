"""
MIT License:
Copyright (c) 2023 Muhammad Umer

Pytorch Lightning module for DAE
"""

from typing import Tuple

import lightning as pl
import torch
import torchmetrics.image as im_metrics
from attack import add_noise_with_snr
from reg_attack import FastGradientSignUntargeted

class LitDAE(pl.LightningModule):
    """
    Denoising Autoencoder (DAE) Pytorch Lightning module.
    """

    def __init__(self, model, cfg, optimizer, criterion, lr_scheduler) -> None:
        """
        Initialize the DAE.

        Args:
            model (nn.Module): DAE model.
            cfg (dict): Configuration dictionary.
            optimizer (optim.Optimizer): Configured optimizer.
            criterion (torchmetrics.Metric): Configured loss function.
            lr_scheduler (lr_scheduler.LRScheduler): Learning rate scheduler.
        """
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.cfg = cfg
        self.loss = criterion

        self.ssim = im_metrics.StructuralSimilarityIndexMeasure()
        self.psnr = im_metrics.PeakSignalNoiseRatio()

        self.epsilon = cfg.get("fgsm_epsilon", 0.1)
        self.attack = cfg.get("fgsm", False)
        self.train_snr_db = cfg.get("snr", None)

        self.pgd = FastGradientSignUntargeted(
            model=self.model,
            epsilon=self.cfg.get("fgsm_epsilon", 0.1),
            alpha=self.cfg.get("pgd_alpha", 0.01),
            min_val=0.0,
            max_val=1.0,
            max_iters=self.cfg.get("pgd_iters", 5),
            _type='linf'
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Training step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Input batch.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss tensor.
        """
        _, img = batch


        if self.train_snr_db is not None:
            img = add_noise_with_snr(img, self.train_snr_db)

        if self.attack:
            print("")
            img = self.pgd.perturb(
                original_images=img,
                labels=None,
                reduction4loss='mean',
                random_start=False
            )

        r_img = self.model(img)
        loss = self.loss(r_img, img)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Validation step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Input batch.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss tensor.
        """
        _, img = batch
        r_img = self.model(img)

        # denoising autoencoder metrics
        loss = self.loss(r_img, img)
        ssim = self.ssim(r_img, img)
        psnr = self.psnr(r_img, img)

        self.log("val_loss", loss, sync_dist=True, prog_bar=True)
        self.log("val_ssim", ssim, sync_dist=True, prog_bar=True)
        self.log("val_psnr", psnr, sync_dist=True, prog_bar=True)

        return loss

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Test step.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): Input batch.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Loss tensor.
        """
        _, img = batch
        # r_img = self.model(img)

        # denoising autoencoder metrics
        # loss = self.loss(r_img, img)
        # ssim = self.ssim(r_img, img)
        # psnr = self.psnr(r_img, img)

        # self.log("test_loss", loss, sync_dist=True, prog_bar=True)
        # self.log("test_ssim", ssim, sync_dist=True, prog_bar=True)
        # self.log("test_psnr", psnr, sync_dist=True, prog_bar=True)
        for snr_db in range(0, 31, 5):  # 测试 SNR 从 0 到 30 每隔 5dB
            noisy_img = self.add_noise_with_snr(img, snr_db)
            r_img = self.model(noisy_img)

            loss = self.loss(r_img, img)
            ssim = self.ssim(r_img, img)
            psnr = self.psnr(r_img, img)

            self.log(f"test_loss_snr{snr_db}", loss, sync_dist=True)
            self.log(f"test_ssim_snr{snr_db}", ssim, sync_dist=True)
            self.log(f"test_psnr_snr{snr_db}", psnr, sync_dist=True)


    @staticmethod
    def add_noise_with_snr(x, snr_db):
        signal_power = x.pow(2).mean()
        snr = 10 ** (snr_db / 10)
        noise_power = signal_power / snr
        noise = torch.randn_like(x) * torch.sqrt(noise_power)
        return torch.clamp(x + noise, 0, 1)

    def configure_optimizers(self) -> Tuple[list, list]:
        """
        Configure optimizers.

        Returns:
            Tuple[list, list]: Optimizers and schedulers.
        """
        return {"optimizer": self.optimizer, "lr_scheduler": self.lr_scheduler}


