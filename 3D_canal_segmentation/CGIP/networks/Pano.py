import math

import numpy as np
import torch
import torch.nn as nn
import torchvision

from networks.ResNet import ResNet18, activation_func
from networks.DLA import DLA34
from networks.HourglassNet import hourglass

from layers.DenseBlock import DenseBlock, TransitionBlock, BasicBlock, BottleneckBlock

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Pano(nn.Module):
    def __init__(self, image_size, patch_size, backbone='resnet', target='offset'):
        super(Pano, self).__init__()
        self._num_teeth = 32
        self._img_size = image_size             # Height, Width
        self._patch_size = patch_size           # Height, Width
        self._target = target

        # layers
        self.upsample = nn.Upsample(size=self._img_size)

        if backbone == 'resnet':
            self.backbone = ResNet18(self._num_teeth * 2)
            self.fmap_channel = 65
            if self._target == 'none':
                self.patch_reg = ResNet18(2, in_channels=self.fmap_channel)
            else:
                self.patch_reg = ResNet18(4, in_channels=self.fmap_channel)
        elif backbone == 'dla':
            self.backbone = DLA34(self._num_teeth * 2)
            self.fmap_channel = 65
            if self._target == 'none':
                self.patch_reg = DLA34(2, in_channels=self.fmap_channel)
            else:
                self.patch_reg = DLA34(4, in_channels=self.fmap_channel)
        elif backbone == 'hourglass':
            self.backbone = hourglass(self._num_teeth)
            self.fmap_channel = 257
            if self._target == 'none':
                self.patch_reg = hourglass(1, in_channels=self.fmap_channel)
            else:
                self.patch_reg = hourglass(2, in_channels=self.fmap_channel)
        else:
            raise NotImplementedError

    def _crop_patch(self, fmap, pts):
        """
        Crop activation map with fixed size
        :param fmap: (N x H x W)
        :param pts: (N x 64)
        :return: (N x )
        """

        y = pts[:, 0::2]
        x = pts[:, 1::2]

        y1 = y - self._patch_size[0] / 2
        y2 = y + self._patch_size[0] / 2
        x1 = x - self._patch_size[1] / 2
        x2 = x + self._patch_size[1] / 2

        grid = torch.zeros(pts.shape[0]*pts.shape[1]//2, 5).cuda()   # (K, 5)
        idx = 0
        for batch_idx in range(pts.shape[0]):
            for center_idx in range(x.shape[1]):
                box = np.array((batch_idx, x1[batch_idx, center_idx].item(), y1[batch_idx, center_idx].item(),
                                x2[batch_idx, center_idx].item(), y2[batch_idx, center_idx].item()))
                grid[idx] = torch.from_numpy(box)
                idx += 1

        # detail for roi_align
        # https://pytorch.org/docs/stable/_modules/torchvision/ops/roi_align.html
        return torchvision.ops.roi_align(fmap, grid, (128, 128))

    def forward(self, x):
        batch_size = x.shape[0]
        input_size = x.shape[-2:]
        y_pts, fmap = self.backbone(x)

        fmap = self.upsample(fmap)
        fmap = torch.cat((x, fmap), dim=1)

        # Resize y_pts to fit in fmap size
        fmap_y_pts = y_pts.clone()
        patches = self._crop_patch(fmap, fmap_y_pts)

        patches, _ = self.patch_reg(patches)

        if self._target == "none":
            bbox = patches[:, 0:2]
            bbox = torch.reshape(bbox, (batch_size, self._num_teeth * 2))

            return y_pts, bbox, _

        else:
            y_offset = patches[:, 0:2]
            y_offset = torch.reshape(y_offset, (batch_size, self._num_teeth * 2))

            bbox = patches[:, 2:4]
            bbox = torch.reshape(bbox, (batch_size, self._num_teeth * 2))

            return y_pts, bbox, y_offset

    def get_patch_size(self):
        return self._patch_size

'''
if __name__ == "__main__":
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from torchsummary import summary
    model = Pano(patch_size=(10, 15))
    model.cuda()

    summary(model, (1, 224, 224))
'''
