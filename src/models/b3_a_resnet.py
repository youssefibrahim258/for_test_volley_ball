from torch import nn
from torchvision import models

class ResNetB3(nn.Module):
    def __init__(self, num_classes=9, dropout_p=0.3):
        super(ResNetB3, self).__init__()

        # Load pretrained ResNet50
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = self.model.fc.in_features

        # Replace the FC layer with Dropout + Linear
        self.model.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)
