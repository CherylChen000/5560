import os, platform
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.gan_model import Generator, Discriminator, weights_init_dcgan

# basic training config
Z_DIM = 100
BATCH_SIZE = 128
EPOCHS = 20
LR_G = 2e-4
LR_D = 2e-4
BETAS = (0.5, 0.999)

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
OUT_DIR = "./artifacts/gan"
os.makedirs(OUT_DIR, exist_ok=True)

# MNIST in [-1,1] to match Tanh
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_set = datasets.MNIST(root="./data", train=True, transform=transform, download=True)

# macOS spawns need workers=0; cuda can use pin_memory
NUM_WORKERS = 0 if platform.system() == "Darwin" else 2
PIN_MEMORY = True if DEVICE == "cuda" else False

loader = torch.utils.data.DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=False
)

# models + init
G = Generator(Z_DIM).to(DEVICE)
D = Discriminator().to(DEVICE)
G.apply(weights_init_dcgan)
D.apply(weights_init_dcgan)

# losses/optims
criterion = nn.BCEWithLogitsLoss()
opt_g = optim.Adam(G.parameters(), lr=LR_G, betas=BETAS)
opt_d = optim.Adam(D.parameters(), lr=LR_D, betas=BETAS)

# fixed noise to track progress
fixed_z = torch.randn(64, Z_DIM, device=DEVICE)

def save_samples(tag: str):
    with torch.no_grad():
        fake = G(fixed_z).cpu()
        grid = utils.make_grid(fake, nrow=8, normalize=True, value_range=(-1, 1))
        utils.save_image(grid, os.path.join(OUT_DIR, f"samples_{tag}.png"))

def main():
    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        G.train(); D.train()
        for real, _ in loader:
            global_step += 1
            real = real.to(DEVICE)
            bsz = real.size(0)

            # D step: real->1, fake->0
            z = torch.randn(bsz, Z_DIM, device=DEVICE)
            fake = G(z).detach()
            logits_real = D(real)
            logits_fake = D(fake)
            real_labels = torch.ones(bsz, 1, device=DEVICE)
            fake_labels = torch.zeros(bsz, 1, device=DEVICE)
            loss_d = criterion(logits_real, real_labels) + criterion(logits_fake, fake_labels)
            opt_d.zero_grad(set_to_none=True); loss_d.backward(); opt_d.step()

            # G step: make D predict 1 on fake
            z = torch.randn(bsz, Z_DIM, device=DEVICE)
            fake = G(z)
            logits_fake = D(fake)
            loss_g = criterion(logits_fake, real_labels)
            opt_g.zero_grad(set_to_none=True); loss_g.backward(); opt_g.step()

            if global_step % 100 == 0:
                print(f"epoch {epoch:02d} step {global_step:05d} | loss_d {loss_d.item():.3f} | loss_g {loss_g.item():.3f}")

        save_samples(f"e{epoch:02d}")
        torch.save({
            "epoch": epoch, "G": G.state_dict(), "D": D.state_dict(),
            "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(), "z_dim": Z_DIM
        }, os.path.join(OUT_DIR, f"ckpt_e{epoch:02d}.pt"))

    print("training done")
    save_samples("final")

if __name__ == "__main__":
    main()
