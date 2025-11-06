import os, platform
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from app.energy_model import EnergyNet, langevin_sample

BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-4
LANGEVIN_STEPS = 40
LANGEVIN_STEP_SIZE = 0.1
LANGEVIN_SIGMA = 0.01

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
OUT_DIR = "./artifacts/energy"
os.makedirs(OUT_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
NUM_WORKERS = 0 if platform.system() == "Darwin" else 2
PIN_MEMORY = True if DEVICE == "cuda" else False

loader = torch.utils.data.DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
)

model = EnergyNet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(1, EPOCHS + 1):
    for i, (x, _) in enumerate(loader):
        x = x.to(DEVICE)
        with torch.no_grad():
            x_neg = langevin_sample(model, n=x.size(0), steps=LANGEVIN_STEPS,
                                     step_size=LANGEVIN_STEP_SIZE, sigma=LANGEVIN_SIGMA, device=DEVICE)
        e_pos = model(x).mean()
        e_neg = model(x_neg).mean()
        loss = e_pos - e_neg

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"epoch {epoch} step {i+1} | loss {loss.item():.4f}")

    with torch.no_grad():
        samples = langevin_sample(model, n=64, steps=LANGEVIN_STEPS,
                                  step_size=LANGEVIN_STEP_SIZE, sigma=LANGEVIN_SIGMA, device=DEVICE)
        grid = utils.make_grid(samples.cpu(), nrow=8, normalize=True, value_range=(-1, 1))
        utils.save_image(grid, os.path.join(OUT_DIR, f"samples_e{epoch:02d}.png"))

    torch.save({"epoch": epoch, "model": model.state_dict()},
               os.path.join(OUT_DIR, f"ckpt_e{epoch:02d}.pt"))

print("EBM training done.")
