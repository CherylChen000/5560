import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    64x64 RGB -> [Conv(3->16,3,1,1) + ReLU + MaxPool(2)]
               -> [Conv(16->32,3,1,1) + ReLU + MaxPool(2)]
               -> Flatten -> FC(32*16*16 -> 100) + ReLU -> FC(100->10)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 64 -> 32

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 32 -> 16
        )
        self.fc1 = nn.Linear(32 * 16 * 16, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)         # (N, 32*16*16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
