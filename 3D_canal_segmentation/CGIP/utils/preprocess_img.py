import os
import numpy as np
import cv2
import math
import pandas as pd
import pydicom
import json
import struct
from scipy.ndimage.interpolation import zoom


def save_as_raw(save_path, array, data_type='B'):
    array = array.flatten()
    temp = data_type * len(array)
    bin = struct.pack(temp, *array)
    f = open(save_path, 'wb')
    f.write(bin)
    f.close()
    return


def eq_hist(img):
    """
    histogram equalizer using CLAHE algorithm
    :param img: numpy image matrix
    :return: equalized img
    """

    # Create an equalizer
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20, 20))

    res = eq.apply(img)

    return res


def crop_center_and_resize(img, size=(224,224)):
    """
    :param img: numpy image matrix(grayscale)
    :return: cropped and resized img
    """

    target_h, target_w = size
    height, width = img.shape
    crop_width = int(height * 1.5)
    margin = (width - crop_width) // 2

    # crop and resize
    img = img[:, margin: margin + crop_width]
    img = cv2.resize(img, dsize=(target_w, target_h))   # dsize has (width, height) form

    # normalize
    img = (np.float64(img) - np.mean(img)) / np.std(img)

    return img


def interpret_coord(coord, size, target_size=(224, 224), use_normalized_coordinates=False):
    """
    change coord, due to center crop, normalize if needed
    :param coord: original coord
    :param size: original height, width of img
    :param size: target height, width of target img
    :param use_normalized_coordinates: if True, normalize coord to [0,1], else [0,224]
    :return: result coord (x, y)
    """
    y, x = coord

    h, w = size
    target_h, target_w = target_size

    crop_width = int(h * 1.5)
    margin = (w - crop_width) // 2
    x -= margin

    # normalize to [0, 1]
    x /= crop_width
    y /= h

    # if not do_normalize, rescale to [0,224]
    if not use_normalized_coordinates:
        x = int(x * target_w)
        y = int(y * target_h)

    return y, x


def restore_coord(coord, size, original_size, use_normalized_coordinates=False):
    """
    Restore original coordinates from resized coordinates.
    :param coord:
    :param original_size:
    :param use_normalized_coordinates:
    :return: Restored coord(x,y)
    """
    original_height, original_width = original_size
    cropped_width = int(original_height * 1.5)
    margin = (original_width - cropped_width) // 2

    y, x = coord

    if not use_normalized_coordinates:
        x /= size[1]
        y /= size[0]

    x = int(x * cropped_width + margin)
    y = int(y * original_height)

    return y, x


def get_meta_dict(meta_path, crop, tooth_list):
    meta_df = pd.read_csv(meta_path)
    meta_dict = {}
    crop_size = crop[1] - crop[0]

    for i, row in meta_df.iterrows():
        name = row['name']
        dcm_depth = eval(row['size'])[0]
        bboxes = []
        for t in tooth_list:
            bbox = eval(row[str(t)])
            if bbox is None:
                bboxes.append([0,0,0])
            else:
                bbox = list(bbox)

                whd = []
                whd.append(int(bbox[1]) - int(bbox[0]))
                whd.append(int(bbox[3]) - int(bbox[2]))
                whd.append(int(bbox[5]) - int(bbox[4]))
                for i in range(3):
                    whd[i] = whd[i] / crop_size

                bboxes.append(whd)

        meta_dict[name] = bboxes

    return meta_dict


def dcm_to_np(dcm_dir, crop, target_width, wc=1024, ww=4096, do_zoom=True):
    dcm_list = os.listdir(dcm_dir)
    dcm_depth = len(dcm_list)
    z_start = crop[4]
    z_end = crop[5]
    crop_width = crop[1] - crop[0]
    ct_array = np.zeros((z_end - z_start, crop_width, crop_width), dtype=np.float32)
    original_size = None

    j = 0
    for i in range(z_start, z_end):
        dcm_file = dcm_list[i]
        dcm_file_path = os.path.join(dcm_dir, dcm_file)
        ds = pydicom.dcmread(dcm_file_path)
        img = np.float32(ds.pixel_array)
        if original_size is None:
            h, w = img.shape
            original_size = (dcm_depth, h, w)

        img = img[crop[2]:crop[3], crop[0]:crop[1]]
        img -= wc - ww / 2
        img /= ww
        img = np.clip(img, 0, 1)
        ct_array[j] = img
        j += 1

    ct_array = np.uint8(ct_array * 255)
    ct_array_zoomed = None
    if do_zoom:
        ct_array_zoomed = zoom(ct_array, target_width / crop_width)

    return ct_array, ct_array_zoomed, original_size


def gaussian3D(shape, sigma=1):
    k, m, n = [(ss - 1.) / 2. for ss in shape]
    z, y, x = np.ogrid[-k:k + 1, -m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y + z * z) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian3D((diameter, diameter, diameter), sigma=diameter / 6)

    x, y, z = int(center[0]), int(center[1]), int(center[2])

    depth, height, width = heatmap.shape

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    inside, outside = min(z, radius), min(depth - z, radius + 1)

    masked_heatmap = heatmap[z - inside:z + outside, y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - inside:radius + outside, radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return


def get_mass_center(coord_list, crop, dcm_depth, exclude_outliers=False, d=128):
    x_offset = crop[0]
    y_offset = crop[2]
    z_offset = dcm_depth + crop[4]

    point_count = len(coord_list) // 3
    mean_x = 0.0
    mean_y = 0.0
    mean_z = 0.0
    for i in range(point_count):
        x = coord_list[3 * i + 0] - x_offset
        y = coord_list[3 * i + 1] - y_offset
        z = coord_list[3 * i + 2] - z_offset
        mean_x = mean_x + (x - mean_x) / (i + 1)
        mean_y = mean_y + (y - mean_y) / (i + 1)
        mean_z = mean_z + (z - mean_z) / (i + 1)

    if exclude_outliers:
        # get rid of outliers
        new_mean_x = 0.0
        new_mean_y = 0.0
        new_mean_z = 0.0
        j = 0
        for i in range(point_count):
            x = coord_list[3 * i + 0] - x_offset
            y = coord_list[3 * i + 1] - y_offset
            z = coord_list[3 * i + 2] - z_offset
            if (mean_x - x)**2 + (mean_y - y)**2 + (mean_z - z)**2 > d**2:
                continue

            new_mean_x = new_mean_x + (x - new_mean_x) / (j + 1)
            new_mean_y = new_mean_y + (y - new_mean_y) / (j + 1)
            new_mean_z = new_mean_z + (z - new_mean_z) / (j + 1)
            j += 1

        mean_x = new_mean_x
        mean_y = new_mean_y
        mean_z = new_mean_z

    return (mean_x, mean_y, mean_z)


def get_bbox(coord_list, crop, dcm_depth, exclude_outliers=True, d=30):
    x_offset = crop[0]
    y_offset = crop[2]
    z_offset = crop[4]

    point_count = len(coord_list) // 3
    mean_x = 0.0
    mean_y = 0.0
    mean_z = 0.0
    if exclude_outliers:
        for i in range(point_count):
            x = coord_list[3 * i + 0]
            y = coord_list[3 * i + 1]
            z = coord_list[3 * i + 2]
            mean_x = mean_x + (x - mean_x) / (i + 1)
            mean_y = mean_y + (y - mean_y) / (i + 1)
            mean_z = mean_z + (z - mean_z) / (i + 1)

    x_min = 1000
    y_min = 1000
    z_min = 1000
    x_max = 0
    y_max = 0
    z_max = 0
    for i in range(point_count):
        x = int(coord_list[3 * i + 0])
        y = int(coord_list[3 * i + 1])
        z = int(coord_list[3 * i + 2])
        if exclude_outliers:
            if (mean_x - x)**2 + (mean_y - y)**2 + (mean_z - z)**2 > d**2:
                continue

        x_min = min(x_min, x)
        y_min = min(y_min, y)
        z_min = min(z_min, z)
        x_max = max(x_max, x)
        y_max = max(y_max, y)
        z_max = max(z_max, z)

    x_min -= x_offset
    x_max -= x_offset
    y_min -= y_offset
    y_max -= y_offset
    z_min -= z_offset
    z_max -= z_offset
    return x_min, x_max, y_min, y_max, z_min, z_max


def get_gaussian_map(json_path, dcm_depth, crop, target_width, r=8):
    heatmap_list = []
    crop_width = crop[1] - crop[0]
    meta_row = []

    with open(json_path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)

        tooth_num = 11
        while tooth_num < 49:
            if tooth_num % 10 == 9:
                tooth_num += 2
                continue

            coords = json_data['annotation']['tooth'][str(tooth_num)]['coordinate'] # coords where mask is painted
            heatmap = np.zeros((target_width, target_width, target_width), dtype=np.float32)
            

            if coords is None:
                meta_row.append('None')
            else:
                x_min, x_max, y_min, y_max, z_min, z_max = get_bbox(coords, crop, dcm_depth)
                meta_row.append((x_min, x_max, y_min, y_max, z_min, z_max))
                x_center = target_width * (x_min + x_max) / (2 * crop_width)
                y_center = target_width * (y_min + y_max) / (2 * crop_width)
                z_center = target_width * (z_min + z_max) / (2 * crop_width)
                draw_gaussian(heatmap, (x_center, y_center, z_center), r)

            heatmap = heatmap * 255
            heatmap = np.uint8(heatmap)
            heatmap_list.append(heatmap)
            tooth_num += 1

    return heatmap_list, meta_row


def create_dist_map_3d(mask, isothetic, diagonal, diagonal3D):
    d, h, w = mask.shape
    dist_map = np.full((d+2, h+2, w+2), 255, dtype=np.uint8)

    a = isothetic
    b = diagonal
    c = diagonal3D
    neighbor_x = [-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0]
    neighbor_y = [-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0]
    neighbor_z = [-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0]
    neighbor_d = [c, b, c, b, a, b, c, b, c, b, a, b, a, 0] #approximate version of euclidean dist
    n_neighbors = len(neighbor_d)

    # first pass
    for k in range(d):
        for j in range(h):
            for i in range(w):
                if mask[k, j, i] == 0:
                    dist_map[k+1, j+1, i+1] = 0
                else:
                    min_dist = dist_map[k+1, j+1, i+1]
                    for n in range(n_neighbors):
                        min_dist = min(min_dist, neighbor_d[n] + dist_map[
                            k+1 + neighbor_z[n], j+1 + neighbor_y[n], i+1 + neighbor_x[n]
                        ])

                    dist_map[k+1, j+1, i+1] = min_dist

    # second pass
    for kk in range(d):
        k = d - kk - 1
        for jj in range(h):
            j = h - jj - 1
            for ii in range(w):
                i = w - ii - 1
                if mask[k, j, i] == 0:
                    dist_map[k+1, j+1, i+1] = 0
                else:
                    min_dist = dist_map[k+1, j+1, i+1]
                    for n in range(n_neighbors):
                        min_dist = min(min_dist, neighbor_d[n] + dist_map[
                            k+1 - neighbor_z[n], j+1 - neighbor_y[n], i+1 - neighbor_x[n]
                        ])

                    dist_map[k+1, j+1, i+1] = min_dist

    return dist_map[1:-1, 1:-1, 1:-1]


def get_3d_mask(json_path, save_path, dcm_dir_name, dcm_depth, crop, do_dist_map=False):
    x_offset = crop[0]
    y_offset = crop[2]
    z_offset = crop[4]
    crop_size = crop[1] - crop[0]

    with open(json_path, 'r', encoding='utf-8') as json_file:
        json_data = json.load(json_file)

        tooth_num = 11
        while tooth_num < 49:
            if tooth_num % 10 == 9:
                tooth_num += 2
                continue

            print(tooth_num)
            coords = json_data['annotation']['tooth'][str(tooth_num)]['coordinate'] # coords where mask is painted
            mask = np.zeros((crop_size, crop_size, crop_size), dtype=np.uint8)
            if coords is not None:
                point_cnt = len(coords) // 3
                for i in range(point_cnt):
                    x = int(coords[3 * i + 0]) - x_offset
                    y = int(coords[3 * i + 1]) - y_offset
                    z = int(coords[3 * i + 2]) - z_offset
                    if x < 0 or y < 0 or z < 0 or crop_size <= x or crop_size <= y or crop_size <= z:
                        continue

                    mask[z, y, x] = 255

            save_as_raw(os.path.join(save_path, 'gt_mask', dcm_dir_name + '_' + str(tooth_num) + '.raw'), mask)

            if do_dist_map:
                if coords is None:
                    dist_map = np.zeros((crop_size, crop_size, crop_size), dtype=np.uint8)
                else:
                    dist_map = create_dist_map_3d(mask, 3, 4, 5)

                save_as_raw(os.path.join(save_path, 'gt_dist_map', dcm_dir_name + '_' + str(tooth_num) + '.raw'), dist_map)

            tooth_num += 1


def preprocess_CT():
    dataset_name = 'cropped_mini'    
    root_path = 'D:/data/healthhub_tooth_CT/'
    rawdata_path = os.path.join(root_path, 'rawdata')
    # data has to be under rawdata_path/dicom/train & val
    # annotation has to be under rawdata_path/annotation/train & val
    
    save_path = os.path.join(root_path, 'gendata', dataset_name)
    meta_path = os.path.join(root_path, 'metadata')
    if not os.path.exists(os.path.join(root_path, 'gendata')):
        os.mkdir(os.path.join(root_path, 'gendata'))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(meta_path):
        os.mkdir(meta_path)

    crop = (151, 601, 50, 500, 0, 450) # crop the original image with (x1, x2, y1, y2, z1, z1)
    target_width = 128 # zoom to fit this size
    crop_width = crop[1] - crop[0]

    train_val = ['train', 'val']
    train_val_count = {'train': 8, 'val': 2} # not random, count from beginning
    out_rows = []
    out_cols = ['name', 'tv', 'size',
        '11', '12', '13', '14', '15', '16', '17', '18',
        '21', '22', '23', '24', '25', '26', '27', '28',
        '31', '32', '33', '34', '35', '36', '37', '38',
        '41', '42', '43', '44', '45', '46', '47', '48']

    for tv in train_val:
        if not os.path.exists(os.path.join(save_path, tv, 'gt_heatmap')):
            os.mkdir(os.path.join(save_path, tv, 'gt_heatmap'))
        if not os.path.exists(os.path.join(save_path, tv, 'gt_debug')):
            os.mkdir(os.path.join(save_path, tv, 'gt_debug'))
        if not os.path.exists(os.path.join(save_path, tv, 'image')):
            os.mkdir(os.path.join(save_path, tv, 'image'))
        if not os.path.exists(os.path.join(save_path, tv, 'image_big')):
            os.mkdir(os.path.join(save_path, tv, 'image_big'))
        if not os.path.exists(os.path.join(save_path, tv, 'gt_center_debug')):
            os.mkdir(os.path.join(save_path, tv, 'gt_center_debug'))
        if not os.path.exists(os.path.join(save_path, tv, 'gt_center_heatmap')):
            os.mkdir(os.path.join(save_path, tv, 'gt_center_heatmap'))
        if not os.path.exists(os.path.join(save_path, tv, 'gt_mask')):
            os.mkdir(os.path.join(save_path, tv, 'gt_mask'))
        if not os.path.exists(os.path.join(save_path, tv, 'gt_dist_map')):
            os.mkdir(os.path.join(save_path, tv, 'gt_dist_map'))

        target_list = os.listdir(os.path.join(root_path, 'dicom', tv))
        for i in range(train_val_count[tv]):
            dcm_dir_name = target_list[i]
            dcm_dir = os.path.join(root_path, 'dicom', tv, dcm_dir_name)            
            dcm_list = os.listdir(dcm_dir)
            dcm_depth = len(dcm_list)
            
            row = [dcm_dir_name, tv]

            print(tv, i, dcm_dir_name)
            ct_array, ct_array_zoomed, original_size = dcm_to_np(dcm_dir, crop, target_width, do_zoom=False)
            row.append(original_size)
            save_as_raw(os.path.join(save_path, tv, 'image_big', dcm_dir_name + '.raw'), ct_array, 'B')
            if ct_array_zoomed is not None:
                save_as_raw(os.path.join(save_path, tv, 'image', dcm_dir_name + '.raw'), ct_array_zoomed, 'B')

            # TODO: need to change gt file format
            json_path = os.path.join(root_path, 'annotation', tv, dcm_dir_name + '.json')
            get_3d_mask(json_path, os.path.join(save_path, tv), dcm_dir_name, dcm_depth, crop, do_dist_map=True)           

            heatmap_list, meta_row = get_gaussian_map(json_path, dcm_depth, crop, target_width)
            row = row + meta_row
            heatmap_sum = np.zeros((target_width, target_width, target_width), dtype=np.uint8())
            for t, hm in enumerate(heatmap_list):
                tooth_num = (t // 8) * 10 + t % 8 + 11
                save_as_raw(os.path.join(save_path, tv, 'gt_center_heatmap', dcm_dir_name + '_' + str(tooth_num) + '.raw'), hm, 'B')
                np.maximum(heatmap_sum, hm, out=heatmap_sum)
            
            save_as_raw(os.path.join(save_path, tv, 'gt_center_debug', dcm_dir_name + '.raw'), heatmap_sum, 'B')
            out_rows.append(row)
        
    out_df = pd.DataFrame(out_rows, columns=out_cols)
    out_df.to_csv(os.path.join(meta_path, dataset_name + '.csv'))


'''
def preprocess_osstem():
    data_path = 'D:/data/osstem_tooth_CT/rawdata/TEST'
    save_path = 'D:/data/osstem_tooth_CT/gendata/cropped'
    crop = (151, 601, 50, 500, 0, 450) # x, x, y, y, z, z
    original_size = (450, 752, 752)
    target_size = (128, 128, 128)
    crop_width = crop[1] - crop[0]
    wc = 1508.5
    ww = 5175

    target_list = os.listdir(os.path.join(data_path))
    for target in target_list:
        dcm_path = os.path.join(data_path, target, 'ct')
        dcm_list = os.listdir(dcm_path)
        ct_array = np.zeros((crop_width, crop_width, crop_width), dtype=np.float32)

        for i, dcm in enumerate(dcm_list):
            dcm_file_path = os.path.join(dcm_path, dcm)
            ds = pydicom.dcmread(dcm_file_path)
            img = np.float32(ds.pixel_array)
            img = img[crop[2]:crop[3], crop[0]:crop[1]]
            img -= wc - ww / 2
            img /= ww
            img = np.clip(img, 0, 1)
            ct_array[crop_width - i - 1] = img

        ct_array = np.uint8(ct_array * 255)
        ct_array = zoom(ct_array, target_size[0] / crop_width)
        save_as_raw(os.path.join(save_path, target + '.raw'), ct_array)
'''


if __name__ == "__main__":
    print('tstart')
    preprocess_CT()
    #preprocess_osstem()
