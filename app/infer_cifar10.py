from io import BytesIO
from typing import Tuple
from PIL import Image
import torch
import torchvision.transforms as transforms

from app.cnn_model import SimpleCNN

CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

_preprocess = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])  # mirrors training

def load_cnn(weights_path="models/cnn_cifar10.pth", device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=10)
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    model.to(device)
    return model

@torch.no_grad()
def predict_image_bytes(model, image_bytes: bytes, device=None) -> Tuple[str, float]:
    device = device or next(model.parameters()).device
    img = Image.open(BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    x = _preprocess(img).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    conf, idx = probs.max(dim=0)
    return CIFAR10_CLASSES[idx.item()], float(conf.item())
