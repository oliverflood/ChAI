import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding="same")
        self.conv2 = nn.Conv2d(64, 128, 3, padding="same")
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(8192, 256)
        self.fc2 = nn.Linear(256, 12)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(-1, 8192)
        x = self.fc1(x)
        x = self.fc2(x)
        return x