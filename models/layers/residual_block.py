import torch
import torch.nn as nn
from models.layers.cnn_block import CNNBlock

class ResidualBlock(nn.Module):
    def __init__(self, in_chs: int, num_repeats: int, use_residual: bool):
        """
        Returns a residual block as a nn.module. It is used by darknet53 and yolov3.
        Inputs:
            >> in_chs: (int) Total number of input channels.
            >> use_residual: (bool) Use of residual block
            >> num_repeats: (int) Number of residual blocks
        """
        super(ResidualBlock, self).__init__()
        self.use_residual = use_residual
        self.layers = nn.ModuleList()
        self.num_repeats = num_repeats
        for _ in range(num_repeats):
            self.layers.append(nn.Sequential(
                CNNBlock(in_chs,    in_chs//2, batch_norm=True, kernel_size=1, stride=1, padding=0),
                CNNBlock(in_chs//2,    in_chs, batch_norm=True, kernel_size=3, stride=1, padding=1)))
        return

    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x if self.use_residual else layer(x)
        return x
