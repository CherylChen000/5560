import torch
import torch.nn as nn


class UNetSmall(nn.Module):
    """Minimal U-Net-like model for 32x32 images (CIFAR-10)."""
    def __init__(self, ch=64):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, ch, 3, 1, 1), nn.SiLU())
        self.enc2 = nn.Sequential(nn.Conv2d(ch, ch * 2, 4, 2, 1), nn.SiLU())
        self.mid = nn.Sequential(nn.Conv2d(ch * 2, ch * 2, 3, 1, 1), nn.SiLU())
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1), nn.SiLU())
        self.out = nn.Conv2d(ch, 3, 3, 1, 1)

    def forward(self, x, t_emb):
        # t_emb kept for API compatibility; not used here
        h1 = self.enc1(x)
        h2 = self.enc2(h1)
        h = self.mid(h2)
        h = self.dec1(h)
        return self.out(h)


class Diffusion:
    """Simplified DDPM process using a linear beta schedule."""
    def __init__(self, model: nn.Module, T: int = 400, beta_start: float = 1e-4,
                 beta_end: float = 0.02, device: str = "cpu"):
        self.model = model
        self.T = T
        self.device = device
        betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def loss(self, x0):
        t = torch.randint(0, self.T, (x0.size(0),), device=self.device)
        noise = torch.randn_like(x0)
        a_bar = self.alpha_bars[t].view(-1, 1, 1, 1)
        xt = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise
        pred = self.model(xt, t)
        return torch.mean((pred - noise) ** 2)

    @torch.no_grad()
    def sample(self, n: int = 64) -> torch.Tensor:
        x = torch.randn(n, 3, 32, 32, device=self.device)
        for ti in reversed(range(self.T)):
            t = torch.full((n,), ti, device=self.device, dtype=torch.long)
            beta = self.betas[ti]
            alpha = self.alphas[ti]
            alpha_bar = self.alpha_bars[ti]
            z = torch.randn_like(x) if ti > 0 else torch.zeros_like(x)
            eps = self.model(x, t)
            x = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_bar)) * eps) + torch.sqrt(beta) * z
            x = x.clamp(-1, 1)
        return x
