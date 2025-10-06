import torch
import torch.nn as nn
from torchvision.models import resnet18

# -------------------------------
# 1️⃣ Define CNN64 architecture
# -------------------------------
class CNN64(nn.Module):
    """
    CNN for 64×64 RGB images:
    Conv(3→16) → ReLU → MaxPool(2)
    Conv(16→32) → ReLU → MaxPool(2)
    Flatten → FC(8192→100) → ReLU → FC(100→10)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),       # 64→32
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),       # 32→16
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 16 * 16, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# -------------------------------
# 2️⃣ Define get_model()
# -------------------------------
def get_model(model_name):
    """
    Return the appropriate model by name.
    Available: 'FCNN', 'CNN', 'EnhancedCNN', 'resnet18'
    """
    if model_name == "CNN":
        return CNN64(num_classes=10)
    elif model_name == "resnet18":
        model = resnet18(weights=None)  # or weights='IMAGENET1K_V1' if allowed
        model.fc = nn.Linear(model.fc.in_features, 10)
        return model
    # TODO: implement FCNN and EnhancedCNN if required
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
