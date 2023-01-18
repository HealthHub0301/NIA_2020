import os
import glob
import numpy as np
import cv2
import torch
import pydicom
from torch.utils.data import Dataset
from scipy.ndimage import zoom
import pandas as pd
import json


class ToothCTDataset(Dataset):
    """
        Dataset for Cone Beam CT Teeth data
    """
    def __init__(self, mode, tooth_list, img_path, gt_path=None, meta_dict=None, type='individual', target_size=(128, 128, 128), do_reg=False):
        self.mode = mode
        self._img_path = img_path
        self._gt_path = gt_path
        self._tooth_list = tooth_list # [11, 12, ... ]
        self._target_size = target_size  # (d, h, w)
        self._output_size_ratio = 0.5
        self.type = type
        self.do_reg = do_reg
        self._img_list = glob.glob(os.path.join(img_path, "*.raw"))
        self.meta_dict = meta_dict

    def __len__(self):
        return len(self._img_list)

    def get_img_filename(self, idx):
        return self._img_list[idx]

    def __getitem__(self, idx):
        # Load img
        img = (np.fromfile(self._img_list[idx], dtype=np.uint8)).reshape(self._target_size)
        img = (np.float64(img) - np.mean(img)) / np.std(img)
        img = torch.FloatTensor(img)
        img = img.unsqueeze(0)

        if self.mode == 'test':
            return img

        # Load gt heatmap file
        img_name = os.path.split(self._img_list[idx])[1]
        hms = None
        if self.type == 'individual':
            hm_list = []
            for tooth_num in self._tooth_list:
                hm = np.fromfile(os.path.join(self._gt_path, img_name[:-4] + '_' + str(tooth_num) + '.raw'), dtype=np.uint8).reshape(self._target_size)
                hm = zoom(hm, self._output_size_ratio)
                hm = np.float64(hm)
                max_val = np.max(hm)
                if max_val > 0:
                    hm /= max_val

                hm_list.append(hm)

            hms = np.array(hm_list, dtype=np.float64)
        elif self.type == 'all':
            hms = np.fromfile(os.path.join(self._gt_path, img_name), dtype=np.uint8).reshape(self._target_size)
            hms = zoom(hms, self._output_size_ratio)
            hms = np.float64(hms)
            max_val = np.max(hms)
            if max_val > 0:
                hms /= max_val

        hms = torch.FloatTensor(hms)

        if self.do_reg:
            bbox = self.meta_dict[img_name[:-4]]
            bbox = torch.FloatTensor(bbox)
            return img, hms, bbox

        return img, hms, []


if __name__=="__main__":
    tooth_list = []
    for i in range(32):
        tooth_num = (i // 8) * 10 + i % 8 + 11
        tooth_list.append(tooth_num)

    root_path = 'D:\\data\\healthhub_tooth_CT'
    prj_name = 'hg_reg'
    dataset_name = 'cropped_100'
    input_size = (128, 128, 128)
    crop = (96, 416, 0, 320, -320, 0) # x, x, y, y, z, z

    from utils.preprocess_img import get_meta_dict
    data_path = os.path.join(root_path, 'gendata', dataset_name)
    meta_path = os.path.join(root_path, 'metadata', dataset_name + '.csv')
    meta_dict = get_meta_dict(meta_path, crop, input_size[0], tooth_list)

    dataset = ToothCTDataset(
        img_path=os.path.join(data_path, 'train', 'image'),
        gt_path=os.path.join(data_path, 'train', 'gt_center_heatmap'),
        meta_dict=meta_dict,
        tooth_list=tooth_list,
        type='individual',
        target_size=input_size
    )

    print(len(dataset))
    print(dataset.__getitem__(0))
