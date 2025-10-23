import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from app.cnn_model import SimpleCNN

EPOCHS = 10
BATCH_SIZE = 128
LR = 1e-3
CLASSES = np.array([
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
])

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_transforms():
    return {
        "train": transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()]),
        "test":  transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    }

def load_data():
    tfm = get_transforms()
    train_ds = torchvision.datasets.CIFAR10(root="./data", train=True,  download=True, transform=tfm["train"])
    test_ds  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm["test"])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False)
    return train_loader, test_loader

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    tot, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        tot += y.size(0)
    return loss_sum / tot, correct / tot

def train():
    torch.manual_seed(42); np.random.seed(42)  
    device = get_device()
    print(f"Using device: {device}")  

    train_loader, test_loader = load_data()
    model = SimpleCNN(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)  
    criterion = nn.CrossEntropyLoss()                  

    best_acc = 0.0
    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * y.size(0)
            total += y.size(0)
            correct += (logits.argmax(1) == y).sum().item()

        train_loss = loss_sum / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            Path("models").mkdir(exist_ok=True, parents=True)
            torch.save({"state_dict": model.state_dict()}, "models/cnn_cifar10.pth")
            print(f"  ✓ Saved best to models/cnn_cifar10.pth (val_acc={best_acc:.4f})")

    print(f"Done in {(time.time()-start)/60:.1f} min. Best val_acc={best_acc:.4f}")

if __name__ == "__main__":
    train()
