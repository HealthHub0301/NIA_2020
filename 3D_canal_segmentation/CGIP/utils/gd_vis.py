import numpy as np
import os
import cv2
import pydicom

if __name__ == '__main__':
    root_path = 'D:/data/temp_img'
    hm1 = cv2.imread(os.path.join(root_path, 'hg', 'hm26.png'), cv2.IMREAD_GRAYSCALE)
    hm2 = cv2.imread(os.path.join(root_path, 'hg', 'hm27.png'), cv2.IMREAD_GRAYSCALE)
    hm3 = cv2.imread(os.path.join(root_path, 'hg', 'hm30.png'), cv2.IMREAD_GRAYSCALE)
    #gd = (hm1 + hm2)
    hm1 = cv2.applyColorMap(hm1, cv2.COLORMAP_JET)
    hm2 = cv2.applyColorMap(hm2, cv2.COLORMAP_JET)
    hm3 = cv2.applyColorMap(hm3, cv2.COLORMAP_JET)

    cv2.imwrite(os.path.join(root_path, 'hg', 'colormap1.png'), hm1)
    cv2.imwrite(os.path.join(root_path, 'hg', 'colormap2.png'), hm2)
    cv2.imwrite(os.path.join(root_path, 'hg', 'colormap3.png'), hm3)
    print(hm1.shape)
