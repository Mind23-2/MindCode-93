import mindspore.nn as nn
from mindspore.ops import operations as P
import numpy as np

class Identify(nn.Cell):
    def __init__(self):
        super(Identify, self).__init__()

    def construct(self, x):
        return x

class SiLU(nn.Cell):
    def __init__(self):
        super(SiLU, self).__init__()
        self.sigmoid = P.Sigmoid()

    def construct(self, x):
        result = x * self.sigmoid(x)
        return result

def convolutional(in_channels,
               out_channels,
               kernel_size,
               stride,
               dilation=1):
    """Get a conv2d batchnorm and relu layer"""
    pad_mode = 'same'
    padding = 0

    return nn.SequentialCell(
        [nn.Conv2d(in_channels,
                   out_channels,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding,
                   dilation=dilation,
                   pad_mode=pad_mode),
         nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5),
         #nn.LeakyReLU(0.1)]
         SiLU()]
    )

class ResidualBlock(nn.Cell):
    """
    CSPDarknet V1 residual block definition.

    Args:
        in_channels: Integer. Input channel.
        filter_num1：Integer. filter_num1
        out_channels: Integer. Output channel.

    Returns:
        Tensor, output tensor.
    Examples:
        ResidualBlock(64,32,64)
    """
    expansion = 4

    def __init__(self,
                 in_channels, out_channels, shortcut=True):

        super(ResidualBlock, self).__init__()
        self.conv1 = convolutional(in_channels, out_channels, kernel_size=1, stride=1)
        self.conv2 = convolutional(out_channels, out_channels, kernel_size=3, stride=1)
        self.add = P.TensorAdd()
        self.short= shortcut

    def construct(self, x):
        identity = x
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        out = self.add(c2, identity)

        if self.short:
            return out
        else:
            return c2

class BottleneckCSP(nn.Cell):
    def __init__(self, block, layer_num, in_channel, out_channel, shortcut=True):
        super(BottleneckCSP, self).__init__()
        self.concat = P.Concat(axis=1)

        # 64 64
        # 128 128
        # 512 512
        out = out_channel // 2
        self.conv1 = convolutional(in_channel, out, 1, 1)
        self.layer1 = self._make_layer(block, layer_num, out, out, shortcut)
        self.conv2 = nn.Conv2d(out, out, kernel_size=1, stride=1, padding=0, dilation=1,pad_mode="same")
        #768
        self.conv3 = nn.Conv2d(in_channel, out, kernel_size=1, stride=1, padding=0, dilation=1,pad_mode="same")
        self.bn1 = nn.BatchNorm2d(out_channel, momentum=0.9, eps=1e-5)
        self.LeakRelu1 = nn.LeakyReLU(0.1)
        self.conv4 = convolutional(out_channel, out_channel, 1, 1)


    def _make_layer(self, block, layer_num, in_channel, out_channel, shortcut=True):
        """
        Make Layer for DarkNet.

        :param block: Cell. DarkNet block.
        :param layer_num: Integer. Layer number.
        :param in_channel: Integer. Input channel.
        :param filter_num1: Integer. filter_num.
        :param out_channel: Integer. Output channel.

        Examples:
            _make_layer(ConvBlock, 1, 64, 32, 64)
        """
        layers = []
        for i in range(0, layer_num):
            darkblk = block(in_channel, out_channel, shortcut)
            layers.append(darkblk)
        return nn.SequentialCell(layers)

    def construct(self, x):
        c1 = self.conv1(x)
        c2 = self.layer1(c1)
        c3 = self.conv2(c2)
        c4 = self.conv3(x)
        c5 = self.concat((c3, c4))
        c6 = self.bn1(c5)
        c7 = self.LeakRelu1(c6)
        c8 = self.conv4(c7)

        return c8


class CSPDarknet(nn.Cell):
    def __init__(self, block, input_shape, detect=False):
        super(CSPDarknet, self).__init__()
        self.detect = detect
        self.concat = P.Concat(axis=1)
        # 12, 32
        self.focus = convolutional(input_shape[0], input_shape[1], 3, 1)
        # CBL-1 32, 64,
        self.conv1 = convolutional(input_shape[1], input_shape[2], 3, 2)
        # CSP-1 64, 64
        self.CSP1 = BottleneckCSP(block, 1 * input_shape[7], input_shape[2], input_shape[2])
        # CBL-2 2, 3 64, 128,
        self.conv2 = convolutional(input_shape[2], input_shape[3], 3, 2)
        # CSP-2 3, 3 128, 128
        self.CSP2 = BottleneckCSP(block, 3 * input_shape[7], input_shape[3], input_shape[3])
        # return CSP2

        #CBL-3 3, 4 128, 256,
        self.conv3 = convolutional(input_shape[3], input_shape[4], 3, 2)

        #CSP-3 4, 4 256, 256
        self.CSP3 = BottleneckCSP(block, 3 * input_shape[7], input_shape[4], input_shape[4])
        # return CSP3

        #CBL-4 4, 5 256, 512,
        self.conv4 = convolutional(input_shape[4], input_shape[5], 3, 2)

        #SPP 5, 4 512, 256,
        self.conv5 = convolutional(input_shape[5], input_shape[4], 1, 1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=13, stride=1, pad_mode='SAME')
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, pad_mode='SAME')
        self.maxpool3 = nn.MaxPool2d(kernel_size=5, stride=1, pad_mode='SAME')
        # 6, 5 1024, 512,
        self.conv6 = convolutional(input_shape[6], input_shape[5], 1, 1)

    def construct(self, x):
        c1 = self.concat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]))
        c2 = self.focus(c1)
        c3 = self.conv1(c2)
        c4 = self.CSP1(c3)
        c5 = self.conv2(c4)
        # out
        c6 = self.CSP2(c5)
        c7 = self.conv3(c6)
        # out
        c8 = self.CSP3(c7)
        c9 = self.conv4(c8)
        c10 = self.conv5(c9)
        m1 = self.maxpool1(c10)
        m2 = self.maxpool2(c10)
        m3 = self.maxpool3(c10)
        c11 = self.concat((m1, m2, m3, c10))
        c12 = self.conv6(c11)

        return c6, c8, c12

    def get_out_channels(self):
        return self.outchannel

