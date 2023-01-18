import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pydicom
from scipy.ndimage.interpolation import zoom
import numpy as np
import torch
from PIL import Image
import io
import json
import pandas as pd
import math
import cv2

from datetime import datetime
import time



from networks.HourglassNet3D import hg_3d
from networks.unet_3d import UNet_3D
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def dcm_to_np(dcm_dir, wc=1024, ww=4096):
    dcm_list = os.listdir(dcm_dir)
    ct_array = []
    for dcm_file_name in dcm_list:
        dcm_file_path = os.path.join(dcm_dir, dcm_file_name)
        ds = pydicom.dcmread(dcm_file_path)
        ct_array.append(np.float32(ds.pixel_array))

    ct_array = np.array(ct_array)
    ct_array -= wc - ww / 2
    ct_array /= ww
    ct_array = np.clip(ct_array, 0, 1)
    ct_array = np.uint8(ct_array * 255)
    return ct_array


def crop_and_zoom(img, crop, target_width):
    img_depth, _, _ = img.shape
    crop_width = crop[1] - crop[0]
    #img = img[img_depth + crop[4]:img_depth + crop[5], crop[2]:crop[3], crop[0]:crop[1]]
    img = img[crop[4]:crop[5], crop[2]:crop[3], crop[0]:crop[1]]
    img = zoom(img, target_width / crop_width)
    return img


def get_center_points(hms, num_class, crop, hm_width=64):
    pts = []
    crop_size = crop[1] - crop[0]
    for c in range(num_class):
        hm = hms[c].squeeze()
        max_idx = torch.argmax(hm)
        max_z = max_idx // (hm_width * hm_width)
        max_y = (max_idx % (hm_width * hm_width)) // hm_width
        max_x = (max_idx % (hm_width * hm_width)) % hm_width

        max_x = int(crop_size * max_x / hm_width)
        max_y = int(crop_size * max_y / hm_width)
        max_z = int(crop_size * max_z / hm_width)

        pts.append((max_x, max_y, max_z))

    return pts


def run_detection(img, model_path, crop, tooth_list):
    # Model
    num_class = len(tooth_list)
    num_feats = 28
    do_reg = True
    separate_head = True
    net = hg_3d(num_stacks=2, num_blocks=1, num_classes=num_class, num_feats=num_feats, in_channels=1,
                hg_depth=3, num_reg=3, do_reg=do_reg, separate_head=separate_head)
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['net'], strict=True)
    net = net.to(device)
    net.eval()

    # Data
    original_size = img.shape
    input_size = (128, 128, 128)
    input = crop_and_zoom(img, crop, input_size[0])
    input = (np.float64(input) - np.mean(input)) / np.std(input)
    input = torch.FloatTensor(input)
    input = input.flip(0)
    input = input.unsqueeze(0).unsqueeze(0) # (1, 1, d, h, w)
    input = input.to(device)

    # Inference
    outputs_bbox = [] # list ob bbox: (x1, x2, y1, y2, z1, z2, max_val)
    with torch.no_grad():
        outputs_hm, outputs_reg = net(input)
        outputs_hm = outputs_hm[-1] # consider only last output
        center_pts = get_center_points(outputs_hm, num_class, crop)

        for c in range(num_class):
            hm = torch.sigmoid(outputs_hm[c].squeeze())
            hm_max = torch.max(hm).item()
            x = center_pts[c][0] + crop[0]
            y = center_pts[c][1] + crop[2]
            #z = center_pts[c][2] + original_size[0] + crop[4]
            z = original_size[0] - (center_pts[c][2] + crop[4])
            w = int(outputs_reg[c][0][0] * original_size[1])
            h = int(outputs_reg[c][0][1] * original_size[1])
            d = int(outputs_reg[c][0][2] * original_size[1])

            outputs_bbox.append((x - w // 2, x + w // 2, y - h // 2, y + h // 2, z - d // 2, z + d // 2, hm_max))
            # print(outputs_bbox[-1])

    return outputs_bbox


def run_segmentation(img, model_path, bbox_list, tooth_list, hm_thresh=0.2, mask_thresh=2):
    # Model
    num_class = len(tooth_list)
    net = UNet_3D(n_class=1)
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['net'], strict=True)
    net = net.to(device)
    net.eval()

    target_size = (128, 64, 64)
    final_output = {}
    D, H, W = img.shape

    # Inference
    for c in range(num_class):
        tooth_num = tooth_list[c]
        x1, x2, y1, y2, z1, z2, val = bbox_list[c]
        #if val < hm_thresh:
        #    final_output[str(tooth_num)] = None
        #    continue

        # add margin to bbox
        # TODO: need to improve segmentation model to get rid of this part
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2
        z_center = (z1 + z2) // 2
        box_width = (x2 - x1 + y2 - y1) // 2
        box_depth = z2 - z1
        if box_depth <= box_width * 2:
            box_depth = box_width * 2

        z_margin = 5
        if tooth_num < 30:
            z_margin *= -1

        new_box_depth = box_depth
        new_box_depth = new_box_depth + 4 - (new_box_depth % 4)
        new_box_width = new_box_depth // 2
        new_bbox = (
            x_center - new_box_width // 2,
            x_center + new_box_width // 2,
            y_center - new_box_width // 2,
            y_center + new_box_width // 2,
            z_center - new_box_depth // 2 + z_margin,
            z_center + new_box_depth // 2 + z_margin
        )

        # Crop
        pad = [-new_bbox[0], new_bbox[1] - W, -new_bbox[2], new_bbox[3] - H, -new_bbox[4], new_bbox[5] - D]
        pad = [max(0, p) for p in pad]
        padded_img = np.pad(img, ((pad[4], pad[5]), (pad[2], pad[3]), (pad[0], pad[1])), 'minimum')
        input = padded_img[
            pad[4] + new_bbox[4]:pad[4] + new_bbox[5],
            pad[2] + new_bbox[2]:pad[2] + new_bbox[3],
            pad[0] + new_bbox[0]:pad[0] + new_bbox[1]
        ]
        input = zoom(input, target_size[0] / new_box_depth)
        input = (np.float64(input) - np.mean(input)) / np.std(input)
        input = torch.FloatTensor(input)
        input = input.to(device)
        if tooth_num < 30: # flip all upper teeth
            input = input.flip(0)

        input = input.unsqueeze(0).unsqueeze(0)  # (1, 1, d, h, w)

        # Inference
        empty_img = np.zeros((D + pad[4] + pad[5], H + pad[2] + pad[3], W + pad[0] + pad[1]), dtype=np.uint8)
        with torch.no_grad():
            output = net(input).squeeze()
            if tooth_num < 30:  # flip all upper teeth
                output = output.flip(0)

            output = output.cpu().numpy()
            output = zoom(output, new_box_depth/ target_size[0])
            output[output <= mask_thresh] = 0
            #output[output > mask_thresh] = 200 + tooth_num
            output[output > mask_thresh] = 1
            #output[output < 0] = 0
            #output[output > 255] = 255
            output = np.uint8(output)

            empty_img[
                pad[4] + new_bbox[4]:pad[4] + new_bbox[5],
                pad[2] + new_bbox[2]:pad[2] + new_bbox[3],
                pad[0] + new_bbox[0]:pad[0] + new_bbox[1]] = output
            empty_img = empty_img[pad[4]:D + pad[4], pad[2]:H + pad[2], pad[0]:W + pad[0]]
            final_output[str(tooth_num)] = empty_img

    return final_output


def get_dice_score(seg_list, json_path, mask_thresh=1.5):
    tooth_count = 0

    dice_sum = 0
    precision_sum = 0
    recall_sum = 0


    with open(json_path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)

        tooth_num = 11
        while tooth_num < 49:
            if tooth_num % 10 == 8:
                tooth_num += 3
                continue

            coords = json_data['annotation']['tooth'][str(tooth_num)]['coordinate']
            if coords is None:
                #print(tooth_num, 'none')
                tooth_num += 1
                continue

            tooth_count += 1
            pred_mask = seg_list[str(tooth_num)]
            pred_area = np.sum(pred_mask)
            intersection_area = 0
            point_cnt = len(coords) // 3
            for i in range(point_cnt):
                x = int(coords[3 * i + 0])
                y = int(coords[3 * i + 1])
                z = int(coords[3 * i + 2])
                #mask[z, y, x] = 255
                if pred_mask[z, y, x] == 1:
                    intersection_area += 1

            #save_as_raw(os.path.join(save_path, 'gt_mask', dcm_dir_name + '_' + str(tooth_num) + '.raw'), mask)
            dice = 2 * intersection_area / (point_cnt + pred_area)
            precision = intersection_area / pred_area
            recall = intersection_area / point_cnt
            #print(tooth_num, dice, precision)
            dice_sum += dice
            precision_sum += precision
            recall_sum += recall

            #dice_min = min(dice_min, dice)
            #dice_max = max(dice_max, dice)
            #precision_min = min(precision_min, precision)
            #precision_max = max(precision_max, precision)

            tooth_num += 1

    #return dice_sum / tooth_count, dice_min, dice_max, precision_sum / tooth_count, precision_min, precision_max

    # caclulate final result, normalize from 2d to 3d
    dice_result = (dice_sum / tooth_count) ** (2 / 3)
    precision_result = (precision_sum / tooth_count) ** (2 / 3)
    recall_result = (recall_sum / tooth_count) ** (2 / 3)

    return dice_result, precision_result, recall_result


def draw_points_on_image_3d(img, box_list, thresh=0.2, color=255, radius=7):
    for bbox in box_list:
        x1, x2, y1, y2, z1, z2, val = bbox
        if val < thresh:
            continue

        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2
        z_center = (z1 + z2) // 2
        for k in range(z_center - radius, z_center + radius + 1):
            r = math.sqrt(radius**2 - (z_center - k)**2)
            cv2.circle(img[k], (x_center, y_center), int(r), color, -1)

    return img


if __name__ == "__main__":
    from utils.visualize import save_as_raw, draw_bbox_3d, draw_seg_result

    #data_path = 'D:/data/healthhub_tooth_CT/rawdata/test/dicom'
    data_path = 'D:/data/osstem_tooth_CT/rawdata/TEST2'
    anno_path = 'D:/data/healthhub_tooth_CT/rawdata/test/annotation'
    experiments_path = 'D:/data/healthhub_tooth_CT/experiments'
    #save_path = os.path.join(experiments_path, 'app_test', 'TTA_log')
    save_path = 'D:/data/osstem_tooth_CT/results'
    detection_model_path = os.path.join(experiments_path, 'hg_final_gd_fix_thresh', 'ckpt-100.pth')
    #segmentation_model_path = os.path.join(experiments_path, 'unet_mini_flip', 'ckpt-300.pth')
    #segmentation_model_path = os.path.join(experiments_path, 'unet_mini_retrain', 'ckpt-160.pth')
    segmentation_model_path = os.path.join(experiments_path, 'unet_all_retrain', 'ckpt-110.pth')

    #crop = (96, 416, 0, 320, -320, 0)
    crop = (151, 601, 50, 500, 0, 450)
    num_tooth_type = 8 # TODO
    num_class = num_tooth_type * 4
    tooth_list = []
    for i in range(num_class):
        tooth_num = (i // num_tooth_type) * 10 + i % num_tooth_type + 11
        tooth_list.append(tooth_num)

    f = open(os.path.join(save_path, 'log.txt'), 'w')

    timestamp = datetime.fromtimestamp(time.time())
    msg = '[' + str(timestamp) + '] Test Start'
    print(msg)
    f.write(msg + '\n')

    final_dice_sum = 0
    final_precision_sum = 0
    final_recall_sum = 0
    final_count = 0

    out_cols = ['tooth_num', 'x1', 'x2', 'y1', 'y2', 'z1', 'z2', 'confidence']

    dcm_list = os.listdir(data_path)
    for dcm_name in dcm_list:
        dcm_dir = os.path.join(data_path, dcm_name, 'ct')
        img = dcm_to_np(dcm_dir)
        #json_path = os.path.join(anno_path, dcm_name + '.json')
        #if not os.path.exists(json_path):
        #    continue

        final_count += 1
        timestamp = datetime.fromtimestamp(time.time())
        msg = '[' + str(timestamp) + '] [Data: ' + dcm_name + '] Loaded Data'
        print(msg)
        f.write(msg + '\n')

        bbox_list = run_detection(img, detection_model_path, crop, tooth_list)
        #img = draw_bbox_3d(img, bbox_list)
        img = np.uint8(img * 0.5)
        img = draw_points_on_image_3d(img, bbox_list)
        #img = np.transpose(img, (2, 0, 1))
        save_as_raw(os.path.join(save_path, dcm_name + '.raw'), img)

        out_rows = []
        for c in range(num_class):
            x1, x2, y1, y2, z1, z2, val = bbox_list[c]
            tooth_num = tooth_list[c]
            out_rows.append([tooth_num, x1, x2, y1, y2, z1, z2, val])

        out_df = pd.DataFrame(out_rows, columns=out_cols)
        out_df.to_csv(os.path.join(save_path, dcm_name + '.csv'))
        continue
        timestamp = datetime.fromtimestamp(time.time())
        msg = '[' + str(timestamp) + '] [Data: ' + dcm_name + '] Detection Complete'
        print(msg)
        f.write(msg + '\n')

        #print('detection done')
        seg_list = run_segmentation(img, segmentation_model_path, bbox_list, tooth_list, mask_thresh=2)
        #print('segmentation done')
        timestamp = datetime.fromtimestamp(time.time())
        msg = '[' + str(timestamp) + '] [Data: ' + dcm_name + '] Segmentation Complete'
        print(msg)
        f.write(msg + '\n')

        #img = draw_bbox_3d(img, bbox_list)
        #img = np.transpose(img, (2, 0, 1))
        #save_as_raw(os.path.join(save_path, 'new', dcm_name + '.raw'), img)
        #img = draw_seg_result(img, seg_list)
        #save_as_raw(os.path.join(save_path, 'seg', dcm_name + '.raw'), img)

        #dice, dice_min, dice_max, precision, precision_min, precision_max = get_dice_score(seg_list, json_path)
        dice, precision, recall = get_dice_score(seg_list, json_path)

        #print(test_count, dcm_name, dice, dice_min, dice_max, precision, precision_min, precision_max)
        #print(test_count, dcm_name, dice, precision, recall)
        timestamp = datetime.fromtimestamp(time.time())
        msg = '[' + str(timestamp) + '] [Data: ' + dcm_name + '] Result: ' + \
              '(Dice = ' + str(dice) + '), ' + \
              '(Precision = ' + str(precision) + '), ' + \
              '(Recall = ' + str(recall) + ')'
        print(msg)
        f.write(msg + '\n')

        final_dice_sum += dice
        final_precision_sum += precision
        final_recall_sum += recall

    timestamp = datetime.fromtimestamp(time.time())
    msg = '[' + str(timestamp) + '] Final Result: ' + \
          '(Dice = ' + str(final_dice_sum / final_count) + '), ' + \
          '(Precision = ' + str(final_precision_sum / final_count) + '), ' + \
          '(Recall = ' + str(final_recall_sum / final_count) + ')'
    print(msg)
    f.write(msg + '\n')
    f.close()


