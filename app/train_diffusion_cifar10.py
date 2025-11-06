import os, platform
import torch
import torch.optim as optim
from torchvision import datasets, transforms, utils
from app.diffusion_model import UNetSmall, Diffusion

BATCH_SIZE = 128
EPOCHS = 20
LR = 2e-4
T_STEPS = 400

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
OUT_DIR = "./artifacts/diffusion"
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

model = UNetSmall().to(DEVICE)
diff = Diffusion(model, T=T_STEPS, device=DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(1, EPOCHS + 1):
    model.train()
    for i, (x, _) in enumerate(loader):
        x = x.to(DEVICE)
        loss = diff.loss(x)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"epoch {epoch} step {i+1} | loss {loss.item():.4f}")

    with torch.no_grad():
        samples = diff.sample(n=64).cpu()
        grid = utils.make_grid(samples, nrow=8, normalize=True, value_range=(-1, 1))
        utils.save_image(grid, os.path.join(OUT_DIR, f"samples_e{epoch:02d}.png"))

    torch.save({"epoch": epoch, "model": model.state_dict()},
               os.path.join(OUT_DIR, f"ckpt_e{epoch:02d}.pt"))

print("Diffusion training done.")
