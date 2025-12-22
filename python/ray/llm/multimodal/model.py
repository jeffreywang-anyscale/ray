import torch
import torch.nn as nn

class QwenVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Output 1024-dim to match text model input dimension
        self.vision_encoder = nn.Linear(10, 1024)

    def forward(self, x):
        return self.vision_encoder(x)

class QwenTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = nn.Linear(1024, 1024)

    def forward(self, x):
        return self.text_encoder(x)