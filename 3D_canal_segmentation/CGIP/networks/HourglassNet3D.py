'''
Hourglass network inserted in the pre-activated Resnet
Use lr=0.01 for current version
(c) YANG, Wei

extended to 3D

'''

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['HourglassNet3D', 'hg_3d']

def _sigmoid(x, e=1e-4):
    return torch.clamp(x.sigmoid_(), min=e, max=1-e)


class Bottleneck3D(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck3D, self).__init__()

        self.bn1 = nn.BatchNorm3d(inplanes)
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out


class Hourglass3D(nn.Module):
    def __init__(self, block, num_blocks, planes, hg_depth):
        super(Hourglass3D, self).__init__()
        self.hg_depth = hg_depth
        self.block = block
        self.hg = self._make_hourglass_3d(block, num_blocks, planes, hg_depth)

    def _make_residual_3d(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes*block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hourglass_3d(self, block, num_blocks, planes, hg_depth):
        hg = []
        for i in range(hg_depth):
            res = []
            for j in range(3):
                res.append(self._make_residual_3d(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual_3d(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hourglass_forward(self, n, x):
        up1 = self.hg[n-1][0](x)
        low1 = F.max_pool3d(x, 2, stride=2)
        low1 = self.hg[n-1][1](low1)

        if n > 1:
            low2 = self._hourglass_forward(n-1, low1)
        else:
            low2 = self.hg[n-1][3](low1)
        low3 = self.hg[n-1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hourglass_forward(self.hg_depth, x)


class HourglassNet3D(nn.Module):
    '''Hourglass model from Newell et al ECCV 2016'''
    def __init__(self, block, in_channels=1, num_stacks=2, num_blocks=1, num_feats=64, num_classes=32, hg_depth=3, num_reg=3, do_reg=True, separate_head=False):
        super(HourglassNet3D, self).__init__()

        self.do_reg = do_reg
        self.separate_head = separate_head
        self.inplanes = num_feats
        self.num_feats = num_feats
        self.hm_size = 64
        self.num_reg = num_reg
        self.num_stacks = num_stacks
        self.num_classes = num_classes
        self.conv1 = nn.Conv3d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm3d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        # build hourglass modules
        ch = self.num_feats*block.expansion
        self.fc1 = self._make_fc_3d(self.inplanes, ch)

        hg, fc, score_, fc_ = [], [], [], []
        head_layers = []
        hm_layers = []
        reg_layers = []

        for i in range(num_stacks):
            hg.append(Hourglass3D(block, num_blocks, self.num_feats, hg_depth))
            fc.append(self._make_fc_3d(ch, ch))
            if self.separate_head:
                for _ in range(num_classes):
                    head_layers.append(self._make_fc_3d(ch, ch))
                    hm_layers.append(nn.Conv3d(ch, 1, kernel_size=1, stride=1))
            else:
                head_layers.append(self._make_fc_3d(ch, ch))
                hm_layers.append(nn.Conv3d(ch, num_classes, kernel_size=1, stride=1))

            if i < num_stacks - 1:
                fc_.append(nn.Conv3d(ch, ch, kernel_size=1, bias=True))
                score_.append(nn.Conv3d(num_classes, ch, kernel_size=1, bias=True))
            elif do_reg:
                if self.separate_head:
                    for _ in range(num_classes):
                        reg_layers.append(nn.Linear(ch, num_reg))
                else:
                    reg_layers.append(nn.Linear(ch, num_reg * num_classes))

        self.hg = nn.ModuleList(hg)
        self.fc = nn.ModuleList(fc)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)
        self.head_layers = nn.ModuleList(head_layers)
        self.hm_layers = nn.ModuleList(hm_layers)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        if do_reg:
            self.reg_layers = nn.ModuleList(reg_layers)

    def _make_residual_3d(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc_3d(self, inplanes, outplanes):
        bn = nn.BatchNorm3d(outplanes)
        conv = nn.Conv3d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
                conv,
                bn,
                self.relu
            )

    def _make_hm_layer_3d(self, inplanes):
        return nn.Sequential(
            nn.Conv3d(inplanes, inplanes, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(inplanes, 1, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc1(x)

        hm_stacks = []
        out_reg = []
        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.fc[i](y)

            hm_list = []
            if self.separate_head:
                for c in range(self.num_classes):
                    y_head = self.head_layers[i * self.num_classes + c](y)
                    hm_out = self.hm_layers[i * self.num_classes + c](y_head)
                    hm_list.append(hm_out)
                    if i == self.num_stacks - 1 and self.do_reg:
                        y_reg = self.avgpool(y_head)
                        y_reg = torch.flatten(y_reg, 1)
                        y_reg = self.reg_layers[c](y_reg)
                        out_reg.append(y_reg)

                hm_stacks.append(hm_list)
            else:
                y_head = self.head_layers[i](y)
                hm_out = self.hm_layers[i](y_head)
                if i == self.num_stacks - 1 and self.do_reg:
                    y_reg = self.avgpool(y_head)
                    y_reg = torch.flatten(y_reg, 1)
                    y_reg = self.reg_layers[0](y_reg)
                    out_reg.append(y_reg)

                hm_stacks.append(hm_out)

            if i < self.num_stacks-1:
                fc_ = self.fc_[i](y)
                if self.separate_head:
                    score_ = self.score_[i](torch.cat(hm_list, dim=1))
                else:
                    score_ = self.score_[i](hm_out)
                x = x + fc_ + score_

        return hm_stacks, out_reg


def hg_3d(**kwargs):
    model = HourglassNet3D(Bottleneck3D, num_stacks=kwargs['num_stacks'], num_blocks=kwargs['num_blocks'],
                num_classes=kwargs['num_classes'], in_channels=kwargs['in_channels'], hg_depth=kwargs['hg_depth'],
                num_reg=kwargs['num_reg'], do_reg=kwargs['do_reg'], num_feats=kwargs['num_feats'], separate_head=kwargs['separate_head'])
    return model


if __name__ == "__main__":
    from torchsummary import summary
    model = hg_3d(num_stacks=2, num_blocks=1, num_classes=32, in_channels=1, hg_depth=3, num_reg=3, do_reg=False, num_feats=64, separate_head=False)
    model.cuda()

    summary(model, (1, 128, 128, 128))
