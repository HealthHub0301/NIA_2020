import os
import glob
import numpy as np
import cv2
import torch
import pydicom
from torch.utils.data import Dataset
from scipy.ndimage.interpolation import zoom
import pandas as pd
import json


class MaskDataset(Dataset):
    """
        Dataset for Cone Beam CT Teeth data Segmentation
    """
    def __init__(self, mode, img_path, gt_path, meta_path, crop, tooth_list, target_size=(128, 64, 64)):
        self.mode = mode
        self._img_path = img_path
        self._gt_path = gt_path
        self._tooth_list = tooth_list # [11, 12, ... ]
        self._target_size = target_size  # (d, h, w)
        self.meta_path = meta_path
        self.crop = crop
        self.crop_size = (crop[1] - crop[0], crop[1] - crop[0], crop[1] - crop[0])
        self.meta_list = []

        meta_df = pd.read_csv(meta_path)
        for i, row in meta_df.iterrows():
            if row['tv'] != mode:
                continue

            name = row['name']
            dcm_depth = eval(row['size'])[0]
            for t in tooth_list:
                bbox = eval(row[str(t)])
                if bbox is None:
                    continue

                new_row = [name, str(t)]
                bbox = list(bbox)
                bbox[0] -= crop[0]
                bbox[1] -= crop[0]
                bbox[2] -= crop[2]
                bbox[3] -= crop[2]
                bbox[4] -= dcm_depth + crop[4]
                bbox[5] -= dcm_depth + crop[4]

                center_x = (bbox[0] + bbox[1]) // 2
                center_y = (bbox[2] + bbox[3]) // 2
                center_z = (bbox[4] + bbox[5]) // 2
                box_width = max(bbox[3] - bbox[2], bbox[1] - bbox[0])
                box_depth = bbox[5] - bbox[4]
                if box_depth <= box_width * 2:
                    box_depth = box_width * 2

                new_box_depth = int(box_depth * 1.2)
                new_box_depth = new_box_depth + 4 - (new_box_depth % 4)
                new_box_width = new_box_depth // 2
                new_bbox = [
                    center_x - new_box_width // 2,
                    center_x + new_box_width // 2,
                    center_y - new_box_width // 2,
                    center_y + new_box_width // 2,
                    center_z - new_box_depth // 2,
                    center_z + new_box_depth // 2
                ]

                if (new_bbox[0] < 0 or new_bbox[2] < 0 or new_bbox[4] < 0 or
                    new_bbox[1] >= self.crop_size[0] or new_bbox[3] >= self.crop_size[0] or new_bbox[3] >= self.crop_size[0]):
                    print(name, t)
                    continue

                new_row.append(new_bbox)
                self.meta_list.append(new_row)

    def __len__(self):
        return len(self.meta_list)

    def get_img_filename(self, idx):
        return self.meta_list[idx][0]

    def get_tooth_num(self, idx):
        return self.meta_list[idx][1]

    def get_bbox(self, idx):
        return self.meta_list[idx][2]

    def get_cropped_img(self, img, bbox):
        img = img[bbox[4]:bbox[5], bbox[2]:bbox[3], bbox[0]:bbox[1]]
        img = zoom(img, self._target_size[0] / (bbox[5] - bbox[4]))
        return img

    def __getitem__(self, idx):
        this_row = self.meta_list[idx]
        this_img_name = this_row[0]
        this_tooth_num = this_row[1]
        this_bbox = this_row[2]

        # Load img and crop
        img = np.fromfile(os.path.join(self._img_path, this_img_name + '.raw'), dtype=np.uint8).reshape(self.crop_size)
        img = self.get_cropped_img(img, this_bbox)
        img = (np.float64(img) - np.mean(img)) / np.std(img)
        img = torch.FloatTensor(img)

        # Load gt mask(or distmap) and crop
        gt = np.fromfile(os.path.join(self._gt_path, this_img_name + '_' + this_tooth_num + '.raw'), dtype=np.uint8).reshape(self.crop_size)
        gt = self.get_cropped_img(gt, this_bbox)
        gt = torch.FloatTensor(gt)

        if int(this_tooth_num) < 30: # flip all upper teeth
            img = img.flip(0)
            gt = gt.flip(0)

        img = img.unsqueeze(0)
        return img, gt


if __name__=="__main__":
    tooth_list = []
    for i in range(32):
        tooth_num = (i // 8) * 10 + i % 8 + 11
        tooth_list.append(tooth_num)

    root_path = 'D:\\data\\healthhub_tooth_CT'
    dataset_name = 'cropped_all'
    crop = (96, 416, 0, 320, -320, 0) # x, x, y, y, z, z

    from utils.preprocess_img import get_meta_dict
    data_path = os.path.join(root_path, 'gendata', dataset_name)
    meta_path = os.path.join(root_path, 'metadata', dataset_name + '.csv')
    mode='train'

    dataset = MaskDataset(
        mode=mode,
        img_path=os.path.join(data_path, mode, 'image_big'),
        gt_path=os.path.join(data_path, mode, 'gt_dist_map'),
        meta_path=meta_path,
        crop=crop,
        tooth_list=tooth_list
    )

    print(len(dataset))
    print(dataset.__getitem__(0)[0].size())
    print(dataset.__getitem__(0)[1].size())
