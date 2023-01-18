import torch
import torch.nn as nn

from networks.ResNet import activation_func

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

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.activate1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.activate2(out)

        return out

class Root(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activate = activation_func('relu')

    def forward(self, *x):
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        x = self.activate(x)

        return x


class Tree(nn.Module):
    def __init__(self, block, in_channels, out_channels, level, stride=1,
                 level_root=False, root_dim=0):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels

        if level_root:
            root_dim += in_channels

        if level==1:
            self.tree1 = block(in_channels, out_channels, stride)
            self.tree2 = block(out_channels, out_channels, 1)
        else:
            self.tree1 = Tree(block, in_channels, out_channels, level-1, stride,
                              root_dim=0)
            self.tree2 = Tree(block, out_channels, out_channels, level-1,
                              root_dim=root_dim+out_channels)

        if level==1:
            self.root = Root(root_dim, out_channels)

        self.root_dim = root_dim
        self.level_root = level_root
        self.level = level
        self.downsample = None
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)

        self.project=None
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual=residual)
        if self.level == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)

        return x


class DLA(nn.Module):
    def __init__(self, in_channels, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, return_levels=False,
                 pool_size=7, linear_root=False):
        super(DLA, self).__init__()

        self.base = self._make_conv_stage(in_channels, channels[0], 1, kernel_size=7, padding=3)

        self.stage1 = self._make_conv_stage(channels[0], channels[0], levels[0])
        self.stage2 = self._make_conv_stage(channels[0], channels[1], levels[1], stride=2)
        self.stage3 = Tree(block, channels[1], channels[2], levels[2], stride=2)
        self.stage4 = Tree(block, channels[2], channels[3], levels[3], stride=2, level_root=True)
        self.stage5 = Tree(block, channels[3], channels[4], levels[4], stride=2, level_root=True)
        self.stage6 = Tree(block, channels[4], channels[5], levels[5], stride=2, level_root=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[5], num_classes)

    def forward(self, x):
        x = self.base(x)

        for i in range(6):
            x = getattr(self, 'stage{}'.format(i+1))(x)

            if i == 2:
                fmap = x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, fmap

    def _make_conv_stage(self, in_channels, out_channels, num_block, kernel_size=3, stride=1, padding=1):
        layers = []

        for i in range(num_block):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(out_channels),
                activation_func('relu')
            ])

            stride = 1
            in_channels = out_channels

        return nn.Sequential(*layers)


def DLA34(num_classes, in_channels=1):
    return DLA(in_channels,
               [1, 1, 1, 2, 2, 1],
               [16, 32, 64, 128, 256, 512],
               num_classes=num_classes)

if __name__ == "__main__":
    from torchsummary import summary

    model = DLA34(4, in_channels=65)
    model.cuda()

    summary(model, (65, 128, 128))
