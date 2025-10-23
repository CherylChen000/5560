# app/gan_model.py
import torch
import torch.nn as nn

# ----- Generator -----
class Generator(nn.Module):
    """
    Input: noise z of shape (B, 100)
    FC -> (B, 128*7*7) -> reshape (B,128,7,7)
    ConvT 128->64, k4 s2 p1 -> (B,64,14,14) + BN + ReLU
    ConvT 64->1,   k4 s2 p1 -> (B,1,28,28) + Tanh
    """
    def __init__(self, z_dim: int = 100):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128 * 7 * 7, bias=False),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(True),

            View((-1, 128, 7, 7)),  # reshape helper

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


# ----- Discriminator -----
class Discriminator(nn.Module):
    """
    Input: image x of shape (B,1,28,28)
    Conv 1->64  k4 s2 p1 -> (B,64,14,14) + LeakyReLU(0.2)
    Conv 64->128 k4 s2 p1 -> (B,128,7,7) + BN + LeakyReLU(0.2)
    Flatten -> Linear(128*7*7 -> 1)  (output logits)
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Linear(128 * 7 * 7, 1)

    def forward(self, x):
        h = self.features(x)
        h = h.view(x.size(0), -1)
        logits = self.classifier(h)  # no sigmoid (we'll use BCEWithLogitsLoss)
        return logits


# ----- helpers -----
class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

def weights_init_dcgan(m):
    """
    DCGAN-style weight init: normal(0, 0.02) for Conv/ConvT/BatchNorm.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.zeros_(m.bias.data)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
