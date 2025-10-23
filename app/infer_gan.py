# app/infer_gan.py
import os
import torch
from torchvision.utils import save_image, make_grid
from gan_model import Generator

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
OUT_DIR = "./artifacts/gan"
os.makedirs(OUT_DIR, exist_ok=True)

def generate(ckpt_path: str, n: int = 64, z_dim: int = 100, out_png: str = None):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    if "z_dim" in ckpt:
        z_dim = ckpt["z_dim"]

    G = Generator(z_dim).to(DEVICE)
    G.load_state_dict(ckpt["G"])
    G.eval()

    with torch.no_grad():
        z = torch.randn(n, z_dim, device=DEVICE)
        imgs = G(z).cpu()
        grid = make_grid(imgs, nrow=int(n**0.5), normalize=True, value_range=(-1, 1))
        out_png = out_png or os.path.join(OUT_DIR, "generated.png")
        save_image(grid, out_png)
        print(f"Saved samples to {out_png}")

if __name__ == "__main__":
    # example:
    # python app/infer_gan.py ./artifacts/gan/ckpt_e20.pt 64
    import sys
    ckpt = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    generate(ckpt, n)
