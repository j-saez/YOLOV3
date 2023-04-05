import torch
import ast
import torch.nn as nn
from models.layers import CNNBlock, ResidualBlock, ScalePrediction

CHS_DIM = 1
OUT_CHS = 0
KERNEL_SZ = 1
STRIDE = 2
NUM_REPEATS = 1

#############
## classes ##
#############

"""
Tuple: cnn_block (out_channels, kernel_size, stride)
List: residual_block ["B", num_repeats]
"S": Scale prediction
"U": Upsampling
"""

class YOLOV3(nn.Module):
    """
    YOLO v3 implementation
    Inputs:
        >> in_chs: (int) Total number of input channels.
        >> num_classes: (int) Total number of classes.
        >> config_file: (str) Path to the network configuration file
    """
    def __init__(self, in_chs: int, num_classes: int, config_file: str):
        super(YOLOV3, self).__init__()
        self.in_chs = in_chs
        self.layers = load_layers_from_configuration(in_chs, num_classes, config_file)
        return

    def forward(self, x):
        outputs = []
        route_connections = []
        for layer in self.layers:

            # After applying a ScalePrediction, we want to continue not from its ouptut, but from the previous
            # layer's output, that is why the continue has been put there.
            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            x = layer(x)

            # At the original implementation, it was seen the route connections were added after a ResidualBlock if num_repeats==8
            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            # We want to concatenate with the last route connection after upsampling
            if isinstance(layer, nn.Upsample):
                x = torch.cat(tensors=(x, route_connections[-1]), dim=CHS_DIM)
                route_connections.pop()

        return outputs

###############
## functions ##
###############

def load_layers_from_configuration(in_chs: int, num_classes: int, config_file: str) -> nn.ModuleList:
    """
    Loads the model from a configuration file.
    Inputs:
        >> in_chs: (int) Total number of input channels.
        >> num_classes: (int) Total number of classes.
        >> config_file: (str) Path to the configuraiton file.
    Outputs:
        >> layers: (nn.ModuleList) Contains all the layers of the model
    """
    layers = nn.ModuleList()
    with open(config_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = ast.literal_eval(line.rstrip('\n'))

            # tuple = ResidualBlock (out_channels, kernel_size, stride)
            if isinstance(line, tuple):
                layers.append(
                    CNNBlock(
                        in_chs,
                        line[OUT_CHS],
                        kernel_size=line[KERNEL_SZ],
                        stride=line[STRIDE],
                        padding=1 if line[KERNEL_SZ]==3 else 0,
                        batch_norm=True))
                in_chs = line[OUT_CHS]

            # list = ResidualBlock (out_channels, kernel_size, stride)
            # ResidualBlock Outputs in_chs, so it does not need to modify in_chs var.
            elif isinstance(line, list):
                layers.append(ResidualBlock(in_chs, line[NUM_REPEATS], use_residual=True ))

            elif isinstance(line, str):
                if line == "S":
                    layers += load_scale_prediction_layer(in_chs, num_classes)
                    in_chs = in_chs // 2
                if line == "U":
                    layers.append(nn.Upsample(scale_factor=2))
                    in_chs = in_chs * 3 # As the concatenation will occur right after the Upsampling layer
                pass
            else:
                config_file = config_file.split('/')[-1]
                raise Exception(f'{line} cannot be interpreted from the {config_file}.')
    return layers

def load_scale_prediction_layer(in_chs: int, num_classes: int) -> nn.ModuleList:
    """
    Returns a Scale prediction layer, which is indicated in the config file by S.
    Inputs:
        >> in_chs: (int) Input channels.
        >> num_classes: (int) Total number of classes.
    Output:
        >> layer: (nn.ModuleList)
    """
    layer = nn.ModuleList()
    layer.append(ResidualBlock(in_chs, num_repeats=1, use_residual=False))
    layer.append(CNNBlock(in_chs, out_chs=in_chs//2, kernel_size=1, batch_norm=True))
    layer.append(ScalePrediction(in_chs//2, num_classes)) 
    return layer
