"""
copied from https://github.com/usuyama/pytorch-unet/blob/master/pytorch_unet.py
extended to 3d
"""
import torch
import torch.nn as nn


def double_conv_3d(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class UNet_3D(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        self.dconv_down1 = double_conv_3d(1, 64)
        self.dconv_down2 = double_conv_3d(64, 128)
        self.dconv_down3 = double_conv_3d(128, 256)
        self.dconv_down4 = double_conv_3d(256, 512)

        self.maxpool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.dconv_up3 = double_conv_3d(256 + 512, 256)
        self.dconv_up2 = double_conv_3d(128 + 256, 128)
        self.dconv_up1 = double_conv_3d(128 + 64, 64)

        self.conv_last = nn.Conv3d(64, n_class, 1)


    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out


if __name__ == "__main__":
    from torchsummary import summary
    model = UNet_3D(n_class=1)
    model.cuda()

    summary(model, (1, 128, 64, 64))
