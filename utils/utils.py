"""
MIT License:
Copyright (c) 2023 Muhammad Umer

Utils for PyTorch Lightning
"""

import sys

import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import TQDMProgressBar
from torch.utils.data import DataLoader
from tqdm import tqdm


def numpy_collate(batch):
    """
    Collate function to use PyTorch datalaoders
    Reference:
    https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
    """
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class SimplifiedProgressBar(TQDMProgressBar):
    """
    Simplified progress bar for non-interactive terminals.
    """

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_predict_tqdm(self):
        bar = super().init_predict_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar

    def init_test_tqdm(self):
        bar = super().init_test_tqdm()
        if not sys.stdout.isatty():
            bar.disable = True
        return bar


def get_mean_std(loader: DataLoader):
    """
    Calculate the mean and standard deviation of the data in the loader.

    Args:
        loader (DataLoader): The DataLoader for which the mean and std are calculated.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The mean and standard deviation of the data in the loader.
    """
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean**2) ** 0.5

    return mean, std


class RayleighChannel(nn.Module):
    """A class used to represent the Rayleigh Channel.

    Attributes:
        sigma (float): a scaling factor for the noise added to the input.
    """

    def __init__(self, sigma):
        """Constructs all the necessary attributes for the Rayleigh Channel object.

        Args:
            sigma (float): a scaling factor for the noise added to the input.
        """
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        """Applies the Rayleigh Channel transformation to the input tensor.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output tensor after adding Rayleigh noise.
        """
        return x + self.sigma * torch.abs(
            (torch.randn_like(x) + 1j * torch.randn_like(x))
            / torch.sqrt(torch.tensor(2.0))
        )
class RicianChannel(nn.Module):
    """A class used to represent the Rician Channel.

    Attributes:
        K (float): Rician factor, the ratio between the power of the direct path and the scattered paths.
        sigma (float): a scaling factor for the noise added to the input.
    """

    def __init__(self, sigma):
        """Constructs all the necessary attributes for the Rician Channel object.

        Args:
            K (float): Rician factor.
            sigma (float): a scaling factor for the noise added to the input.
        """
        
        super().__init__()
        self.K = 2
        self.sigma = sigma

    def forward(self, x):
        """Applies the Rician Channel transformation to the input tensor.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: output tensor after adding Rician noise.
        """

        device = x.device
        dtype = x.dtype
        # Generate the direct path component (LOS)
        # s = torch.sqrt(self.K / (1 + self.K)) * torch.ones_like(x)
        # s = torch.sqrt(torch.tensor(self.K / (1 + self.K), device=x.device, dtype=x.dtype)) * torch.ones_like(x)
        
        # # Generate the scattered path component (NLOS)
        # noise = torch.randn_like(x) + 1j * torch.randn_like(x)
        
        # # Combine both components and add to input
        # rician_noise = s + torch.sqrt(1 / (1 + self.K)) * noise
        
        # return x + self.sigma * torch.abs(rician_noise)

        K_tensor = torch.tensor(self.K / (1 + self.K), device=device, dtype=dtype)
        s = torch.sqrt(K_tensor) * torch.ones_like(x)

        # Generate the scattered path component (NLOS)
        noise = torch.randn_like(x) + 1j * torch.randn_like(x)

        # Combine both components and add to input
        fading_ratio = torch.tensor(1 / (1 + self.K), device=device, dtype=dtype)
        rician_noise = s + torch.sqrt(fading_ratio) * noise

        return x + self.sigma * torch.abs(rician_noise)
    

class ChannelModel(nn.Module):
    def __init__(self, mode='rayleigh', snr_db=25):
        super().__init__()
        self.mode = mode
        self.snr_db = snr_db

    def forward(self, x):
        if self.mode == 'awgn':
            return self._awgn(x)
        elif self.mode == 'rayleigh':
            return self._rayleigh(x)
        else:
            return x  # no channel

    def _awgn(self, x):
        power = x.pow(2).mean()
        snr = 10 ** (self.snr_db / 10)
        noise_power = power / snr
        noise = torch.randn_like(x) * torch.sqrt(noise_power)
        return x + noise

    def _rayleigh(self, x):
        fading = torch.randn_like(x) * (1.0 / 2**0.5)
        return self._awgn(x * fading)
