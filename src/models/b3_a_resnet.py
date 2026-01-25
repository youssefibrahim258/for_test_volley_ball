# resnetb3_resnet50_light.py
from torch import nn
from torchvision import models

class ResNetB3(nn.Module):
    def __init__(self, num_classes=9, dropout_p=0.3, freeze_layers=True):
        super(ResNetB3, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes)
        )
        if freeze_layers:
            for layer in [self.model.layer1, self.model.layer2]:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x):
        return self.model(x)
