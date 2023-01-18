import torch
import torch.nn as nn

def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=False)],
        ['none', nn.Identity()]
    ])[activation]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, activation='relu'):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activate1 = activation_func(activation)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activate2 = activation_func(activation)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            # Downsample
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=self.expansion * out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.activate1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activate2(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, activation='relu'):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activate1 = activation_func(activation)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activate2 = activation_func(activation)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=self.expansion * out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)
        self.activate3 = activation_func(activation)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            # Downsample
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        """

        :param x: Input tensor
        :return: Output tensor
        """
        out = self.activate1(self.bn1(self.conv1(x)))
        out = self.activate2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.activate3(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, in_channels=1):
        super(ResNet, self).__init__()

        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.activate = activation_func('relu')
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = self._make_layer(block, 64, num_blocks[0])
        self.conv3 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.conv4 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.conv5 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, in_channels, num_block, stride=1):
        strides = [stride] + [1] * (num_block-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, in_channels, stride))
            self.in_planes = in_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.activate(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        fmap = x
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, fmap


def ResNet18(num_classes, in_channels=1):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, in_channels=in_channels)


def ResNet50(num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


if __name__ == "__main__":
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from torchsummary import summary
    model = ResNet18(32*2)
    model.cuda()

    summary(model, (1, 512, 768))

    import torchvision.models as models

    summary(models.resnet18().cuda(), (3,224,224))
