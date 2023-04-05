import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_chs: int, out_chs: int, batch_norm: bool,  **kwargs):
        """
        Returns a convolutional layer as a nn.module. It is used by darknet53 and yolov3.
        Inputs:
            >> in_chs: (int) Total number of input channels.
            >> out_chs: (int) Total number of output channels.
            >> batch_norm: (bool) Apply batch normalization.
            >> kwargs: Will be the kernel_size, stride and padding values when they are different that the default ones in Conv2d
        """
        super(CNNBlock, self).__init__()
        model = nn.Sequential()
        model.append(nn.Conv2d(in_chs, out_chs, bias=not batch_norm, **kwargs))
        if batch_norm:
            model.append(nn.BatchNorm2d(out_chs))
        model.append(nn.LeakyReLU(0.1))
        self.model = model
        return

    def forward(self, x):
        return self.model(x)
