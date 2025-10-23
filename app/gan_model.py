import torch
import torch.nn as nn

# small reshape helper so I can keep nn.Sequential
class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
    def forward(self, x):
        return x.view(*self.shape)

# generator: FC -> (128,7,7) -> upsample to (1,28,28)
class Generator(nn.Module):
    def __init__(self, z_dim: int = 100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128 * 7 * 7, bias=False),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(True),
            View((-1, 128, 7, 7)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
    def forward(self, z):
        return self.net(z)

# discriminator: downsample to 7x7 and classify with a linear head
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Linear(128 * 7 * 7, 1)
    def forward(self, x):
        h = self.features(x).view(x.size(0), -1)
        return self.classifier(h)

# dcgan-style init for convs / bns / linears
def weights_init_dcgan(m):
    n = m.__class__.__name__
    if "Conv" in n or "ConvTranspose" in n:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in n:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.zeros_(m.bias.data)
    elif "Linear" in n:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
