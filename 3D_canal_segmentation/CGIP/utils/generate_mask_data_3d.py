import os
import sys
import math
import cv2
import numpy as np
import json
import pandas as pd


def is_true(s):
    if 'T' in s or 't' in s:
        return True
    return False


def create_dist_map(mask):
    h, w = mask.shape
    dist_map = np.full((h+2, w+2), 255, dtype=np.uint8)
    neighbor_x = [-1, 0, 1, -1]
    neighbor_y = [-1, -1, -1, 0]
    neighbor_d = [4, 3, 4, 3]

    # first pass
    for j in range(h):
        for i in range(w):
            if mask[j, i] == 0:
                dist_map[j+1, i+1] = 0
            else:
                min_dist = dist_map[j+1, i+1]
                for n in range(4):
                    min_dist = min(min_dist,
                                   neighbor_d[n] + dist_map[j+1 + neighbor_y[n], i+1 + neighbor_x[n]])

                dist_map[j+1, i+1] = min_dist

    # second pass
    for jj in range(h):
        j = h - jj - 1
        for ii in range(w):
            i = w - ii - 1
            if mask[j, i] == 0:
                dist_map[j+1, i+1] = 0
            else:
                min_dist = dist_map[j+1, i+1]
                for n in range(4):
                    min_dist = min(min_dist,
                                   neighbor_d[n] + dist_map[j+1 - neighbor_y[n], i+1 - neighbor_x[n]])

                dist_map[j+1, i+1] = min_dist

    return dist_map


def get_mask(txt_file_path, H, W):
    tooth_num_list = []
    bbox_list = []
    mask_list = []
    dist_map_list = []
    df = pd.read_csv(txt_file_path, header=None)

    for i, row in df.iterrows():
        tooth_num = int(row[0])
        tooth_exists = is_true(str(row[1]))
        points = []

        x_max = y_max = -math.inf
        x_min = y_min = math.inf
        j = 0
        while j < 16:
            x = int(row[j+3]) + W // 2
            y = int(row[j+4]) + H // 2
            points.append([x, y])
            x_max = max(x_max, x)
            y_max = max(y_max, y)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            j += 2

        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2
        points = np.array(points)

        mask = np.zeros((H, W), dtype=np.uint8)
        if tooth_exists:
            mask = cv2.fillPoly(mask, [points], 255)

        tooth_num_list.append(tooth_num)
        bbox_list.append([x_min, y_min, x_max, y_max])
        mask_list.append(mask)

    return tooth_num_list, bbox_list, mask_list


def main():
    source_path = None # path to det result files(json)
    numbered = True # if source is det && numbered result, else False
    margin = 10 # size of margin from aabb

    data_path = 'D:\\osstem_pano_data\\'
    raw_path = data_path + 'rawdata\\anonymous\\'
    group_list = ['train', 'val', 'test']

    for group in group_list:
        output_path = data_path + 'gendata\\mask_data\\' + group
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # create mask data from gt txt files
        if source_path is None:
            cur_path = os.path.join(raw_path, group)
        else:
            cur_path = os.path.join(source_path, group)

        file_list = os.listdir(cur_path)
        for f in file_list:
            if f[-3:] != 'txt' and f[-4:] != 'json':
                continue

            print(f)
            img_id = (f.split('.')[0]).split('_')[0]
            img_file_path = os.path.join(raw_path, group, img_id + '.jpg')
            txt_file_path = os.path.join(raw_path, group, img_id + '.txt')

            img = cv2.imread(img_file_path, cv2.IMREAD_GRAYSCALE)
            H, W = img.shape
            # get 32 masks and 32 chamfer distance maps
            tooth_num_list, bbox_list, mask_list = get_mask(txt_file_path, H, W)

            if source_path is None:
                # create mask data for all 32 gt
                for i, (x1, y1, x2, y2) in enumerate(bbox_list):
                    tooth_num = tooth_num_list[i]
                    img_patch = img[y1 - margin : y2 + margin, x1 - margin : x2 + margin]
                    mask_patch = mask_list[i][y1 - margin : y2 + margin, x1 - margin : x2 + margin]

                    dist_map_patch = create_dist_map(mask_patch)
                    dist_map_patch = dist_map_patch[1:-1, 1:-1]

                    cv2.imwrite(os.path.join(output_path, img_id + '_' + str(tooth_num) + '_crop.jpg'), img_patch)
                    cv2.imwrite(os.path.join(output_path, img_id + '_' + str(tooth_num) + '_mask.jpg'), mask_patch)
                    cv2.imwrite(os.path.join(output_path, img_id + '_' + str(tooth_num) + '_chamfer.jpg'), dist_map_patch)

            else:
                with open(os.path.join(cur_path, f)) as det_json:
                    dets = json.load(det_json)

                for d in dets:
                    # TODO: when det is 1-class(not numbered)
                    tooth_num = int(d['tooth_num'])
                    tooth_center = d['center']
                    bbox_w, bbox_h = d['bbox_wh']

                    x1 = tooth_center[0] - bbox_w // 2
                    x2 = tooth_center[0] + bbox_w // 2
                    y1 = tooth_center[1] - bbox_h // 2
                    y2 = tooth_center[1] + bbox_h // 2

                    img_patch = img[y1 - margin : y2 + margin, x1 - margin : x2 + margin]
                    mask_patch = mask_list[i][y1 - margin : y2 + margin, x1 - margin : x2 + margin]

                    dist_map_patch = create_dist_map(mask_patch)
                    dist_map_patch = dist_map_patch[1:-1, 1:-1]

                    cv2.imwrite(os.path.join(output_path, img_id + '_' + str(tooth_num) + '.jpg'), img_patch)
                    cv2.imwrite(os.path.join(output_path, img_id + '_' + str(tooth_num) + '_mask.jpg'), mask_patch)
                    cv2.imwrite(os.path.join(output_path, img_id + '_' + str(tooth_num) + '_chamfer.jpg'), dist_map_patch)


if __name__ == '__main__':
    main()
