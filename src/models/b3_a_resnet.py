from torch import nn
from torchvision import models

class ResNetB3(nn.Module):
    def __init__(self, num_classes=8):
        super(ResNetB3, self).__init__()

        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.in_features = self.model.fc.in_features

        self.dropout = nn.Dropout(p=0.3)
        self.model.fc = nn.Sequential(
            self.dropout,
            nn.Linear(self.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)
