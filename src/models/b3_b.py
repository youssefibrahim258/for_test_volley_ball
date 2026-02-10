import torch
import torch.nn as nn
import torch.nn.functional as F

class Stage2Classifier(nn.Module):
    """
    Stage 2 classifier for B3:
    - Input: pooled feature vector from ResNet50 (2048,)
    - Output: 8 classes (clip-level action classification)
    """

    def __init__(self, input_dim=2048, hidden_dim=1024, num_classes=8, dropout=0.5):
        super().__init__()

        # Hidden layer 1
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu=nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Output layer
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
