import torch

def fgsm_attack(model, loss_fn, images, labels, epsilon):
    images.requires_grad = True
    outputs = model(images)
    loss = loss_fn(outputs, images)
    model.zero_grad()
    loss.backward()
    perturbed_images = images + epsilon * images.grad.sign()
    return torch.clamp(perturbed_images, 0, 1)


def add_noise_with_snr(x: torch.Tensor, snr_db: float) -> torch.Tensor:
    signal_power = x.pow(2).mean()
    snr = 10 ** (snr_db / 10)
    noise_power = signal_power / snr
    noise = torch.randn_like(x) * torch.sqrt(noise_power)
    return torch.clamp(x + noise, 0, 1)
