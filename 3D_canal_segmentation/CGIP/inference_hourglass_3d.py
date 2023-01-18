import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import cv2
import struct
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.ToothCTDataset import ToothCTDataset
from utils.preprocess_img import get_meta_dict
from networks.ResNet import ResNet18
from networks.HourglassNet3D import hg_3d
from utils.visualize import draw_points_on_image_3d, save_as_raw, get_center_points, draw_bbox_3d
from scipy.ndimage.interpolation import zoom
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_2d_heatmap(hm_3d, num_class, hm_width=64, target_width=320):
    hm_2d_list = []
    for c in range(num_class):
        hm = hm_3d[c].squeeze()
        hm = torch.sigmoid(hm)
        hm = hm.cpu().numpy() # (d, h, w)
        hm = np.max(hm, 0)
        hm_max = np.max(hm)
        hm = np.uint8(hm * 255 / hm_max)
        hm = zoom(hm, target_width / hm_width)
        hm_2d_list.append(hm)

    return np.array(hm_2d_list)

if __name__ == "__main__":
    # dirs
    root_path = 'D:\\data\\healthhub_tooth_CT'
    #root_path = 'D:\\data\\osstem_tooth_CT'
    #prj_name = 'hg_32_focal'
    #prj_name = 'hg_centernet_32'
    prj_name = 'hg_final_gd_fix_thresh'

    #dataset_name = 'cropped_100'
    dataset_name = 'cropped'
    #target_img = '00125907_20200914'

    #data_path = os.path.join(root_path, 'gendata', dataset_name)
    #meta_path = os.path.join(root_path, 'metadata', dataset_name + '.csv')

    model_path = os.path.join(root_path, 'experiments', prj_name)
    output_path = os.path.join(root_path, 'experiments', prj_name, 'results')
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    input_size = (128, 128, 128)
    heatmap_size = (64, 64, 64)
    #crop = (96, 416, 0, 320, -320, 0)  # x, x, y, y, z, z
    crop = (151, 601, 50, 500, 0, 450)  # x, x, y, y, z, z
    num_tooth_type = 8
    num_class = num_tooth_type * 4
    tooth_list = []
    for i in range(num_class):
        tooth_num = (i // num_tooth_type) * 10 + i % num_tooth_type + 11
        tooth_list.append(tooth_num)

    do_reg = True
    num_feats = 28
    separate_head = True

    # Model
    net = hg_3d(num_stacks=2, num_blocks=1, num_classes=num_class, num_feats=num_feats, in_channels=1,
                hg_depth=3, num_reg=3, do_reg=do_reg, separate_head=separate_head)

    # load checkpoint
    checkpoint_epoch = 100
    checkpoint = torch.load(os.path.join(model_path, 'ckpt-' + str(checkpoint_epoch) + '.pth'))
    net.load_state_dict(checkpoint['net'], strict=True)
    net = net.to(device)
    net.eval()

    # test dataset
    tv = 'val'
    #meta_dict = get_meta_dict(meta_path, crop, tooth_list)
    meta_dict = None
    '''test_dataset = ToothCTDataset(        
        mode='test',
        img_path=os.path.join(data_path, tv, 'image'),
        gt_path=os.path.join(data_path, tv, 'gt_center_heatmap'),
        meta_dict=meta_dict,
        tooth_list=tooth_list,
        type='individual',
        target_size=input_size,
        do_reg=do_reg
    )'''
    test_dataset = ToothCTDataset('test', tooth_list=tooth_list, img_path='D:/data/osstem_tooth_CT/gendata/cropped')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    print("Dataset Done")

    out_cols = ['tooth_num', 'x1', 'x2', 'y1', 'y2', 'z1', 'z2', 'val']
    original_size = 512

    with torch.no_grad():
        for batch_idx, inputs in enumerate(test_loader):
        #for batch_idx, (inputs, hm_targets, reg_targets) in enumerate(test_loader):

            out_rows = []


            #hm_targets = hm_targets.to(device)
            #reg_targets = reg_targets.to(device)
            img_path = test_dataset.get_img_filename(batch_idx)

            img_name = os.path.split(img_path)[1]
            #if img_name[:-4] != target_img:
            #    continue

            print(img_name)

            inputs = inputs.to(device)
            outputs_hm, outputs_reg = net(inputs)
            outputs_hm = outputs_hm[-1]#.squeeze()
            #outputs_reg = outputs_reg[0].squeeze()

            #print(torch.min(outputs_hm), torch.max(outputs_hm))
            #break
            #outputs_hm *= 255
            #outputs_hm = np.uint8(outputs_hm.cpu().numpy())
            #for i in range(num_class):
            #    save_as_raw(os.path.join(output_path, img_name[:-4] + '_' + str(tooth_list[i]) + '.raw'), outputs_hm[i])
            #break

            #hm_2d = get_2d_heatmap(outputs_hm, num_class)
            #hm_temp = hm_2d[26:30, :, :]
            #print(hm_temp.shape)
            #hm_temp = np.max(hm_temp, 0)
            #for iii in range(24, 32):
            #    cv2.imwrite('D:/data/temp_img/hm' + str(iii) + '.png', hm_2d[iii])

            #break


            center_pts = get_center_points(outputs_hm, num_class, crop)

            raw_img = np.fromfile(img_path, dtype=np.uint8).reshape((128, 128, 128))
            raw_img = np.uint8(raw_img * 0.5)
            #raw_img = draw_points_on_image_3d(raw_img, center_pts, crop)
            draw_bbox_3d(raw_img, num_class, center_pts, outputs_reg, crop, tooth_list, os.path.join(output_path, 'osstem_bbox', img_name))
            #raw_img = np.transpose(raw_img, (2,0,1))
            #save_as_raw(os.path.join(output_path, img_name), raw_img)


            #row = [img_name[:-4]]

            for c in range(num_class):
                tooth_num = tooth_list[c]
                hm = torch.sigmoid(outputs_hm[c].squeeze())
                hm_max = torch.max(hm).item()

                #hm_max = center_pts[c][3]
                x = center_pts[c][0] + crop[0]
                y = center_pts[c][1] + crop[2]
                z = center_pts[c][2] + crop[4]# + original_size
                z = crop[5] - z
                w = int(outputs_reg[c][0][0] * original_size)
                h = int(outputs_reg[c][0][1] * original_size)
                d = int(outputs_reg[c][0][2] * original_size)
                w = max(w, h)
                h = max(w, h)
                #w = int(outputs_reg[3 * c + 0] * original_size)
                #h = int(outputs_reg[3 * c + 1] * original_size)
                #d = int(outputs_reg[3 * c + 2] * original_size)
                #row.append((x - w // 2, x + w // 2, y - h // 2, y + h // 2, z - d // 2, z + d // 2, hm_max))
                row = [tooth_num, x - w // 2, x + w // 2, y - h // 2, y + h // 2, z - d // 2, z + d // 2, hm_max]

                '''
                w_margin = 30
                h_margin = 50
                d_margin = 50
                # print(x, y, z, w, h, d)
                w += w_margin
                h += h_margin
                d += d_margin
                z = 450 - z

                row.append(x - w // 2)
                row.append(x + w // 2)
                row.append(y - h // 2)
                row.append(y + h // 2)
                row.append(z - d // 2)
                row.append(z + d // 2)

                #print(tooth_num, row[3], hm_max)
                out_rows.append(row)
                '''

                out_rows.append(row)

            out_df = pd.DataFrame(out_rows, columns=out_cols)
            out_df.to_csv(os.path.join(output_path, 'results_' + img_name[:-4] + '.csv'))
