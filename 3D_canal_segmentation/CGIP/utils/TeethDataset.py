import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from utils.preprocess_img import eq_hist, crop_center_and_resize, interpret_coord

class TeethDataset(Dataset):
    """
        Dataset for Teeth data
    """
    def __init__(self, path, target_size=(512, 768)):
        self._path = path
        self._target_size = target_size  # (h, w)

        self._img_list = glob.glob(os.path.join(path, "*.jpg"))
        self._txt_list = glob.glob(os.path.join(path, "*.txt"))

        assert(len(self._img_list) == len(self._txt_list))

    def __len__(self):
        return len(self._img_list)

    def __getitem__(self, idx):
        # Load img
        img = cv2.imread(self._img_list[idx], cv2.IMREAD_GRAYSCALE)

        height, width = img.shape

        # Preprocess img
        img = eq_hist(img)

        # Crop and resize img
        img = crop_center_and_resize(img, size=self._target_size)

        # Load text file
        with open(self._txt_list[idx], 'r') as txt_f:
            content = txt_f.readlines()

            teeth = {}
            teeth_existance = {}
            teeth_aabb = {}
            for line in content:

                # Skip for empty line
                if line == "":
                    continue
                teeth_data = line.split(',')

                if len(teeth_data) < 2:
                    continue

                teeth_num = int(teeth_data[0].strip())

                # # Calculate alveolar position
                # alveolar_x = width // 2 + int(teeth_data[-2])
                # alveolar_y = height // 2 + int(teeth_data[-1])
                # alveolar = (alveolar_x, alveolar_y)
                #
                # teeth[teeth_num] = alveolar

                # Existance
                teeth_existance[teeth_num] = (1 if teeth_data[1].strip() == "True" else 0)

                # Load 8 points polygon
                # AABB
                polygon_min_x = float("inf")
                polygon_min_y = float("inf")
                polygon_max_x = 0.0
                polygon_max_y = 0.0

                for i in range(8):
                    point_x = width // 2 + int(teeth_data[3+2*i])
                    point_y = height // 2 + int(teeth_data[4+2*i])

                    if point_x < polygon_min_x:
                        polygon_min_x = point_x
                    if point_y < polygon_min_y:
                        polygon_min_y = point_y
                    if point_x > polygon_max_x:
                        polygon_max_x = point_x
                    if point_y > polygon_max_y:
                        polygon_max_y = point_y

                # points represent as (y, x) form
                center = ((polygon_min_y + polygon_max_y) / 2, (polygon_min_x + polygon_max_x) / 2)
                teeth[teeth_num] = center
                teeth_aabb[teeth_num] = (interpret_coord((polygon_min_y, polygon_min_x),
                                                         (height, width),
                                                         target_size=self._target_size),
                                         interpret_coord((polygon_max_y, polygon_max_x),
                                                         (height, width),
                                                         target_size=self._target_size))

        targets = sorted(teeth)
        targets = [interpret_coord(teeth[key], (height, width), target_size=self._target_size) for key in targets]
        targets = np.array(targets)

        # targets_existance = sorted(teeth_existance)
        # targets_existance = [teeth_existance[key] for key in targets_existance]

        targets_aabb = sorted(teeth_aabb)
        targets_aabb = [teeth_aabb[key] for key in targets_aabb]

        assert(len(targets) == 32)

        # Make a flatten array
        targets = targets.flatten()

        img = torch.FloatTensor(img)
        img = img.unsqueeze(0)
        targets = torch.FloatTensor(targets)
        #targets_existance = torch.FloatTensor(targets_existance)
        targets_aabb = torch.FloatTensor(targets_aabb)

        return img, targets, targets_aabb

    def get_img_filename(self, idx):
        return self._img_list[idx]


if __name__=="__main__":
    dataset = TeethDataset(path='E:/Data/Teeth-Pano/test')
    print(len(dataset))
    print(dataset.__getitem__(0))
    print(dataset.__getitem__(0)[1])
    print("here")
    print(dataset.__getitem__(0)[2].shape)

    import matplotlib.pyplot as plt

    for i in range(len(dataset)):
        img = dataset.__getitem__(i)[0]
        img = img.squeeze(0)
        img_name = dataset.get_img_filename(i).split('\\')[-1]
        plt.imsave(img_name, img)

        # print(img.shape)
        # cv2.imshow("test", img.numpy() )
        # cv2.waitKey()

    from utils.visualize import visualize
    #visualize_result('E:/Data/Teeth-Pano/test/000012.jpg', '.', dataset.__getitem__(0)[1], is_normalized=False)
    visualize('E:/Data/Teeth-Pano/test/000012.jpg', '.',
              dataset.__getitem__(0)[1].numpy(), dataset.__getitem__(0)[2].numpy(), size=(512, 768),
              show_center=True)

    # for d in dataset:
    #     print(d)
