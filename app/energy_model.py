import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyNet(nn.Module):
    """Small CNN that maps 3x32x32 CIFAR-10 images to a scalar energy."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        h = self.features(x).view(x.size(0), -1)
        return self.fc(h)


def langevin_sample(model, n=64, steps=80, step_size=0.1, sigma=0.01, device="cpu"):
    """
    Langevin dynamics in pixel space.
    Enables gradient only for x, keeping model params frozen.
    """
    # temporarily freeze model parameters
    was_requires_grad = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)

    x = torch.randn(n, 3, 32, 32, device=device).clamp_(-1, 1)
    x.requires_grad_(True)

    for _ in range(steps):
        # compute ∇E(x)
        with torch.enable_grad():
            energy = model(x).sum()
            grad_x = torch.autograd.grad(
                energy, x, create_graph=False, retain_graph=False
            )[0]

        # update: move against ∇E(x) and add noise
        with torch.no_grad():
            x -= step_size * grad_x
            x += sigma * torch.randn_like(x)
            x.clamp_(-1, 1)

        x.requires_grad_(True)

    # restore model param grad flags
    for p, w in zip(model.parameters(), was_requires_grad):
        p.requires_grad_(w)

    return x.detach()
