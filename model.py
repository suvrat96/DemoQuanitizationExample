import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub


def conv_relu_bn_block(in_channels, out_channels, kernel_size, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )


class CustomCNN(nn.Module):
    def __init__(self, args):
        super(CustomCNN, self).__init__()
        self.args = args

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        head_channel_sum = 0
        in_channels = 3
        self.trunk_blocks = nn.ModuleList()
        for i_block, out_channels in enumerate(args.model_trunk_num_filters):
            self.trunk_blocks.append(conv_relu_bn_block(in_channels, out_channels, 3))
            in_channels = out_channels
            if i_block in self.args.model_pool_layer_indices:
                head_channel_sum += out_channels

        in_channels = head_channel_sum
        self.head_blocks = nn.ModuleList()
        for i_block, out_channels in enumerate(args.model_head_num_filters):
            self.head_blocks.append(conv_relu_bn_block(in_channels, out_channels, 3))
            in_channels = out_channels

        self.clf = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.concat_pool = nn.AdaptiveAvgPool2d(32)

        self.model_pool_layer_indices = self.args.model_pool_layer_indices

        self.q_cat = torch.nn.quantized.FloatFunctional()


    def forward(self, batch):
        concat = []
        out = batch
        out = self.quant(out)
        for i_layer, layer in enumerate(self.trunk_blocks):
            out = layer(out)
            if i_layer in self.model_pool_layer_indices:
                out = self.pool(out)
                concat.append(self.concat_pool(out))

        out = self.q_cat.cat(concat, dim=1)
        for layer in self.head_blocks:
            out = layer(out)

        out = self.clf(out)

        out = self.dequant(out)
        return out


    def fuse_model(self):
        for block in self.trunk_blocks:
            torch.quantization.fuse_modules(block, ['0', '1', '2'], inplace=True)

        for block in self.head_blocks:
            torch.quantization.fuse_modules(block, ['0', '1', '2'], inplace=True)

        torch.quantization.fuse_modules(self.clf, ['0', '1'], inplace=True)
