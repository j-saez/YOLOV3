import torch
import torch.nn as nn
from models.layers.cnn_block import CNNBlock

ANCHORS_PER_CELL = 3
TOTAL_BBOX_ELEMENTS = 5 #[object probability, x, y, w, h]

class ScalePrediction(nn.Module):
    def __init__(self, in_chs: int, num_classes: int):
        """
        TODO
        Inputs:
            >> in_chs: (int) Number of input channels.
            >> num_classes: (int) Number of classes in the data.
            >> num_anchors: (int) Number of anchors boxes per cell.
        """
        super(ScalePrediction, self).__init__()
        out_chs = ANCHORS_PER_CELL * (num_classes+TOTAL_BBOX_ELEMENTS)
        self.prediction = nn.Sequential(
            CNNBlock(  in_chs, 2*in_chs, batch_norm=True,  kernel_size=3, stride=1, padding=1),
            CNNBlock(2*in_chs,  out_chs, batch_norm=False, kernel_size=1, stride=1, padding=0))
        self.num_classes = num_classes
        return

    def forward(self, x):
        B,CHS,H,W = x.size()
        x = self.prediction(x)
        x = x.view(B, ANCHORS_PER_CELL, self.num_classes+TOTAL_BBOX_ELEMENTS, H, W)
        x = x.permute(0,1,3,4,2)
        return x

###########
## tests ##
###########

if __name__ == '__main__':
    print('Tesing ScalePrediction')
    batch_size = 8
    in_chs = 64
    input = torch.rand(batch_size, in_chs, 13, 13)
    layer = ScalePrediction(in_chs, num_classes=10)
    output = layer(input)
