import os
import torch
from torchvision.utils import make_grid, save_image
from io import BytesIO
from app.diffusion_model import UNetSmall, Diffusion

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
OUT_DIR = "./artifacts/diffusion"
os.makedirs(OUT_DIR, exist_ok=True)


@torch.no_grad()
def sample_png_bytes(ckpt_path: str | None = None, n: int = 64, T: int = 400) -> bytes:
    """Load a diffusion checkpoint and return a PNG grid of generated samples."""
    if ckpt_path is None:
        paths = sorted([p for p in os.listdir(OUT_DIR) if p.startswith("ckpt_e") and p.endswith(".pt")])
        assert paths, "No diffusion checkpoints found."
        ckpt_path = os.path.join(OUT_DIR, paths[-1])

    state = torch.load(ckpt_path, map_location=DEVICE)
    net = UNetSmall().to(DEVICE)
    net.load_state_dict(state["model"])
    net.eval()

    diff = Diffusion(net, T=T, device=DEVICE)
    imgs = diff.sample(n=n).cpu()
    grid = make_grid(imgs, nrow=int(n ** 0.5), normalize=True, value_range=(-1, 1))

    buf = BytesIO()
    save_image(grid, buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()
