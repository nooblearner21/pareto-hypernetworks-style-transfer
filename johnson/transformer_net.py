import torch.nn as nn

class StyleTransferNet(nn.Module):
    def __init__(self):
        super(StyleTransferNet, self).__init__()

        """
        Downsampling Layers
        """
        self.DownsampleBlock = nn.Sequential(
            ConvLayer(3, 32, 9, 1),
            nn.InstanceNorm2d(32, affine=True),
            ConvLayer(32,64,3,2),
            nn.ReLU(),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            ConvLayer(64,128,3,2),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(),
        )

        """
        Residual Layers
        """
        self.ResBlock = nn.Sequential(
            ResLayer(128, kernel=3, stride=1),
            ResLayer(128, kernel=3, stride=1),
            ResLayer(128, kernel=3, stride=1),
            ResLayer(128, kernel=3, stride=1),
            ResLayer(128, kernel=3, stride=1),
        )

        """
        Upsampling Layers
        """
        self.UpsampleBlock = nn.Sequential(
            UpsampleLayer(128,64, kernel=3, stride=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            UpsampleLayer(64,32, kernel=3, stride=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
        )

        """
        Output
        """
        self.output = ConvLayer(32,3,9,1)

    def forward(self, x):
        x = self.DownsampleBlock(x)
        x = self.ResBlock(x)
        x = self.UpsampleBlock(x)
        out = self.output(x)
        return out


"""
Downsampling ConvLayer
"""
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        #Uses reflection padding to avoid border artifcats and disturbance
        self.reflection_pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x = self.reflection_pad(x)
        out = self.conv1(x)
        return out


"""
Residual Blocks With Instance Normalization
"""
class ResLayer(nn.Module):
    def __init__(self, channels, kernel, stride):
        super(ResLayer, self).__init__()

        self.conv1 = ConvLayer(channels, channels, kernel, stride)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.relu1 = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel, stride)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x + residual


"""
Upsampling Layer
Based on https://distill.pub/2016/deconv-checkerboard/
to avoid checkerboard upsampling 'effect' by using nearest neighbour interpolation
and a convoloution layer with dimension-preserving padding
"""
class UpsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride):
        super(UpsampleLayer, self).__init__()

        padding = kernel // 2
        self.upsample = nn.Upsample(scale_factor=2)
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel, stride)

    def forward(self, x):
        x= self.upsample(x)
        x = self.reflection_pad(x)
        x = self.conv1(x)
        return x