import os
import torch
from torchvision.utils import make_grid, save_image
from io import BytesIO
from app.energy_model import EnergyNet, langevin_sample

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
OUT_DIR = "./artifacts/energy"
os.makedirs(OUT_DIR, exist_ok=True)


@torch.no_grad()
def sample_png_bytes(ckpt_path: str | None = None, n: int = 64) -> bytes:
    """Load an energy-based model checkpoint and return a PNG image grid as bytes."""
    if ckpt_path is None:
        paths = sorted([p for p in os.listdir(OUT_DIR) if p.startswith("ckpt_e") and p.endswith(".pt")])
        assert paths, "No EBM checkpoints found."
        ckpt_path = os.path.join(OUT_DIR, paths[-1])

    state = torch.load(ckpt_path, map_location=DEVICE)
    net = EnergyNet().to(DEVICE)
    net.load_state_dict(state["model"])
    net.eval()

    imgs = langevin_sample(net, n=n, device=DEVICE)
    grid = make_grid(imgs.cpu(), nrow=int(n ** 0.5), normalize=True, value_range=(-1, 1))

    buf = BytesIO()
    save_image(grid, buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()
