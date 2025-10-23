import os, torch
from torchvision.utils import make_grid, save_image
from app.gan_model import Generator

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
OUT_DIR = "./artifacts/gan"
os.makedirs(OUT_DIR, exist_ok=True)

# simple CLI/helper to load a checkpoint and write a grid png
def generate(ckpt_path: str, n: int = 64, z_dim: int = 100, out_png: str | None = None, seed: int | None = None):
    state = torch.load(ckpt_path, map_location=DEVICE)
    if "z_dim" in state:
        z_dim = state["z_dim"]

    G = Generator(z_dim).to(DEVICE)
    G.load_state_dict(state["G"])
    G.eval()

    if seed is not None:
        torch.manual_seed(seed)

    with torch.no_grad():
        z = torch.randn(n, z_dim, device=DEVICE)
        imgs = G(z).cpu()
        grid = make_grid(imgs, nrow=int(n ** 0.5), normalize=True, value_range=(-1, 1))
        out_png = out_png or os.path.join(OUT_DIR, "generated.png")
        save_image(grid, out_png)
        print(f"saved {out_png}")

if __name__ == "__main__":
    import sys
    ckpt = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else None
    generate(ckpt, n=n, seed=seed)
