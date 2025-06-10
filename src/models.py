import timm
from torch import nn


class TorchModel(nn.Module):
    def __init__(self, encoder, pretrained, output):
        super(TorchModel, self).__init__()
        self.model = timm.create_model(encoder, pretrained, num_classes=output)

    def forward(self, x):
        return self.model(x)
