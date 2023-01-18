import numpy as np
import cv2
import os
import math
import struct

import torch
from utils.preprocess_img import restore_coord
from eval import get_intersection, get_area

TRAIN_INPUT_SIZE = 224
COLORS = {
    # Colors must be BGR order
    'aliceblue': (255, 248, 240),
    'aqua': (229, 211, 190),
    'black': (0, 0, 0),
    'blue': (255, 0, 0),
    'blueviolet': (89, 17, 54),
    'chartreuse': (0, 255, 128),
    'lightskyblue': (250, 206, 135),
    'white': (255, 255, 255),
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'yellow': (0, 255, 255),
    'cyan': (255, 255, 0),
    'magenta': (255, 0, 255)
}


def visualize_compare(img_path, output_path, gt_points, gt_boxes, y_points, y_offsets, y_boxes, size, is_normalized=True):
    """
        draw 32 points and aabb boxes on image
        :param img_path: path of original image
        :param output_path: path to save result image
        :param result: list that contains 32 points(len: 64)
        :param is_normalized: if True, result values are normalized to [0,1], else [0,224]
    """
    img_id = os.path.split(img_path)[1][:-4]
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    original_height, original_width = img.shape[:2]

    teeth_cnt = 32

    box_name_list = []
    for i in range(teeth_cnt):
        gt_y = gt_points[2 * i]
        gt_x = gt_points[2 * i + 1]
        gt_y, gt_x = restore_coord((gt_y, gt_x), size, (original_height, original_width))
        gt_points[2 * i] = gt_y
        gt_points[2 * i + 1] = gt_x

        min_point, max_point = gt_boxes[i]
        min_point = restore_coord(min_point, size, (original_height, original_width))
        max_point = restore_coord(max_point, size, (original_height, original_width))

        gt_boxes[i] = (min_point, max_point)

        y = y_offsets[2 * i]
        x = y_offsets[2 * i + 1]
        y, x = restore_coord((y, x), size, (original_height, original_width))
        y_offsets[2 * i] = y
        y_offsets[2 * i + 1] = x

        min_point, max_point = y_boxes[i]
        min_point = restore_coord(min_point, size, (original_height, original_width))
        max_point = restore_coord(max_point, size, (original_height, original_width))
        y_boxes[i] = (min_point, max_point)

        tooth_num = (i // 8) * 10 + (i % 8) + 11
        box_name_list.append(str(tooth_num))

    # GT boxes
    draw_bounding_boxes_on_image(img, gt_boxes, color="chartreuse", box_name_list=box_name_list)
    draw_bounding_boxes_on_image(img, y_boxes, color="lightskyblue", box_name_list=box_name_list)

    #
    #
    # # draw y_origin
    # for i in range(32):
    #     x = y_points[2 * i]
    #     y = y_points[2 * i + 1]
    #     if not is_normalized:
    #         x /= y_size[0]
    #         y /= y_size[1]
    #
    #     x = int(x * new_W + margin)
    #     y = int(y * H)
    #
    #     cv2.circle(img, (x, y), 3, COLORS['cyan'], -1)

    cv2.imwrite(os.path.join(output_path, img_id + '_compare.jpg'), img)


def visualize(img_path, output_path, points, boxes, gt_boxes, size, show_center=False, is_gt=False):
    """

    :param img_path:
    :param output_path:
    :param points:
    :param boxes:
    :param gt_boxes:
    :param size:
    :param show_center:
    :param is_gt:
    :return:
    """
    img_id = os.path.split(img_path)[1][:-4]
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    original_height, original_width = img.shape[:2]

    teeth_cnt = 32

    assert(len(points) == teeth_cnt*2)
    assert(len(boxes) == teeth_cnt)

    box_name_list = []
    iou_list = []
    for i in range(teeth_cnt):
        y = points[2 * i]
        x = points[2 * i + 1]
        y, x = restore_coord((y, x), size, (original_height, original_width))
        points[2 * i] = y
        points[2 * i + 1] = x

        min_point, max_point = boxes[i]
        gt_min_point, gt_max_point = gt_boxes[i]
        intersect = get_intersection((min_point[0], min_point[1], max_point[0], max_point[1]),
                                     (gt_min_point[0], gt_min_point[1], gt_max_point[0], gt_max_point[1]))

        infer_area = get_area((min_point[0], min_point[1], max_point[0], max_point[1]))
        gt_area = get_area((gt_min_point[0], gt_min_point[1], gt_max_point[0], gt_max_point[1]))
        intersection_area = get_area(intersect)

        iou = intersection_area / (infer_area + gt_area - intersection_area)

        min_point = restore_coord(min_point, size, (original_height, original_width))
        max_point = restore_coord(max_point, size, (original_height, original_width))
        boxes[i] = (min_point, max_point)

        tooth_num = (i // 8) * 10 + (i % 8) + 11
        box_name_list.append(str(tooth_num))
        iou_list.append(iou)

    # Inference boxes
    draw_bounding_boxes_on_image(img, boxes, color="lightskyblue",
                                 box_name_list=box_name_list,
                                 iou_list=iou_list,
                                 show_center=show_center)

    if is_gt is True:
        cv2.imwrite(os.path.join(output_path, img_id + '_gt.jpg'), img)
    else:
        cv2.imwrite(os.path.join(output_path, img_id + '_result.jpg'), img)


def visualize_pts(img_path, output_path, size, **kwargs):
    img_id = os.path.split(img_path)[1][:-4]
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    original_height, original_width = img.shape[:2]

    teeth_cnt = 32

    for key, value in kwargs.items():
        if key in ['org_pts']:
            pts = []
            for i in range(teeth_cnt):
                y = value[2 * i]
                x = value[2 * i + 1]
                y, x = restore_coord((y, x), size, (original_height, original_width))
                pts.append((y, x))

            draw_points_on_image(img, pts, color='lightskyblue')

    cv2.imwrite(os.path.join(output_path, img_id + '_pts.jpg'), img)


def draw_bounding_boxes_on_image(img,
                                 boxes,
                                 color='red',
                                 thickness=4,
                                 box_name_list=(),
                                 iou_list=(),
                                 show_center=False):

    if len(box_name_list) == 0:
        box_name_list = [''] * len(boxes)

    assert(len(boxes) == len(box_name_list))

    for i, (box, box_name) in enumerate(zip(boxes, box_name_list)):
        min_point, max_point = box
        ymin, xmin = min_point
        ymax, xmax = max_point

        iou = None
        if iou_list:
            iou = iou_list[i]

        # For flipped box
        dir = 1
        if int(box_name) > 30:
            dir = -1

        draw_bounding_box_on_image(img, xmin, ymin, xmax, ymax, box_name, iou_value=iou, dir=dir,
                                   color=color, thickness=thickness, show_center=show_center)

    return img


def draw_bounding_box_on_image(img,
                               xmin,
                               ymin,
                               xmax,
                               ymax,
                               box_name,
                               iou_value=None,
                               color='red',
                               thickness=4,
                               dir=1,
                               show_center=False):
    """
    Draw a bounding box to an image.
    :param img: image matrix from openCV
    :param xmin: xmin of bbox
    :param ymin: ymin of bbox
    :param xmax: xmax of bbox
    :param ymax: ymax of bbox
    :param color: color of bounding box edges. Default is red.
    :param thickness: thickness of bounding box edges.
    :return: image has a bounding box
    """
    img_height, img_width = img.shape[:2]

    font_scale = 2
    font = cv2.FONT_HERSHEY_PLAIN
    text_width, text_height = cv2.getTextSize(box_name, font, font_scale, thickness=1)[0]

    if show_center is True:
        cv2.circle(img, (int((xmin + xmax) / 2), int((ymin + ymax) / 2)), 5, COLORS[color], -1)

    # Bbox drawing
    cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), COLORS[color], thickness=thickness)


    if dir == 1:
        cv2.rectangle(img, (int(xmin - thickness/2), int(ymin - text_height - 10)), (int(xmin + text_width + 2), int(ymin)),
                      COLORS[color], cv2.FILLED)
        cv2.putText(img, box_name, (int(xmin), int(ymin - 2)), font, font_scale, COLORS['black'], thickness=2)

        if iou_value:
            cv2.putText(img, "%d%%"%(iou_value * 100), (int(xmin), int(ymin + 25)), font, font_scale, COLORS['black'], thickness=2)
    else:
        cv2.rectangle(img, (int(xmin - thickness / 2), int(ymax + text_height + 10)),
                      (int(xmin + text_width + 2), int(ymax)),
                      COLORS[color], cv2.FILLED)
        cv2.putText(img, box_name, (int(xmin), int(ymax + 28)), font, font_scale, COLORS['black'], thickness=2)

        if iou_value:
            cv2.putText(img, "%d%%" % (iou_value * 100), (int(xmin), int(ymax + 55)), font, font_scale,
                        COLORS['black'], thickness=2)


    return img


def draw_points_on_image(img,
                         pts,
                         color='red'):
    for y, x in pts:
        draw_point_on_image(img, x, y, color=color)

    return img


def draw_point_on_image(img,
                        x,
                        y,
                        color='red',
                        radius=5):
    cv2.circle(img, (x, y), radius, COLORS[color], -1)
    return img


def draw_points_on_image_3d(img, pts, crop, color=255, radius=3):
    crop_size = crop[1] - crop[0]
    img_size = img.shape[0]
    for x, y, z, _ in pts:
        #x -= crop[0]
        #y -= crop[2]
        #z -= 512 + crop[4]
        x = int(img_size * x / crop_size)
        y = int(img_size * y / crop_size)
        z = int(img_size * z / crop_size)

        for k in range(z - radius, z + radius + 1):
            if k < 0 or img_size <= k:
                continue
            r = math.sqrt(radius**2 - (z - k)**2)
            cv2.circle(img[k], (x, y), int(r), color, -1)

    return img


def save_as_raw(save_path, array, data_type='B'):
    array = array.flatten()
    temp = data_type * len(array)
    bin = struct.pack(temp, *array)
    f = open(save_path, 'wb')
    f.write(bin)
    f.close()
    return


def save_cropped_teeth(img_path, output_path, boxes, size):
    """

    :param img_path:
    :param output_path:
    :param boxes:
    :param size: (h, w)
    :return:
    """
    img_id = os.path.split(img_path)[1][:-4]
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    original_height, original_width = img.shape[:2]


    # Create Directory
    dir_name = "teeth"
    dir = os.path.join(output_path, dir_name)
    if not os.path.exists(dir):
        os.mkdir(dir)

    img_dir = os.path.join(dir, img_id)
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)

    teeth_cnt = 32

    for i in range(teeth_cnt):

        min_point, max_point = boxes[i]
        min_point = restore_coord(min_point, size, (original_height, original_width))
        max_point = restore_coord(max_point, size, (original_height, original_width))

        cropped_img = img[int(min_point[0]):int(max_point[0]), int(min_point[1]):int(max_point[1])]

        tooth_num = (i // 8) * 10 + (i % 8) + 11

        cv2.imwrite(os.path.join(img_dir, str(tooth_num) + '.jpg'), cropped_img)


def get_center_points(hms, n_class, crop, hm_width=64):
    pts = []
    crop_size = crop[1] - crop[0]
    for c in range(n_class):
        hm = hms[c].squeeze()
        max_val = torch.max(hm).item()
        max_idx = torch.argmax(hm)
        max_z = max_idx // (hm_width * hm_width)
        max_y = (max_idx % (hm_width * hm_width)) // hm_width
        max_x = (max_idx % (hm_width * hm_width)) % hm_width

        max_x = int(crop_size * max_x / hm_width)
        max_y = int(crop_size * max_y / hm_width)
        max_z = int(crop_size * max_z / hm_width)

        pts.append((max_x, max_y, max_z, max_val))

    return pts


def draw_line_3d(raw_img, start, v, n, color=255):
    img_size = raw_img.shape
    x, y, z = start
    v_x, v_y, v_z = v
    for i in range(n):
        if 0 <= x and x < img_size[2] and 0 <= y and y < img_size[1] and 0 <= z and z < img_size[0]:
            raw_img[z][y][x] = color

        x += v_x
        y += v_y
        z += v_z


def draw_bbox_3d_from_center(raw_img, n_class, center_pts, whd, crop):
    crop_size = crop[1] - crop[0]
    img_size = raw_img.shape[0]

    for c in range(n_class):
        x = center_pts[c][0]
        y = center_pts[c][1]
        z = center_pts[c][2]
        x = int(img_size * x / crop_size)
        y = int(img_size * y / crop_size)
        z = int(img_size * z / crop_size)
        w = int(whd[c][0][0] * img_size)
        h = int(whd[c][0][1] * img_size)
        d = int(whd[c][0][2] * img_size)
        w_margin = 12
        h_margin = 20
        d_margin = 20
        w += w_margin
        h += h_margin
        d += d_margin

        draw_line_3d(raw_img, (x - w // 2, y - h // 2, z - d // 2), (1, 0, 0), w)
        draw_line_3d(raw_img, (x - w // 2, y + h // 2, z - d // 2), (1, 0, 0), w)
        draw_line_3d(raw_img, (x - w // 2, y - h // 2, z + d // 2), (1, 0, 0), w)
        draw_line_3d(raw_img, (x - w // 2, y + h // 2, z + d // 2), (1, 0, 0), w)

        draw_line_3d(raw_img, (x - w // 2, y - h // 2, z - d // 2), (0, 1, 0), h)
        draw_line_3d(raw_img, (x + w // 2, y - h // 2, z - d // 2), (0, 1, 0), h)
        draw_line_3d(raw_img, (x - w // 2, y - h // 2, z + d // 2), (0, 1, 0), h)
        draw_line_3d(raw_img, (x + w // 2, y - h // 2, z + d // 2), (0, 1, 0), h)

        draw_line_3d(raw_img, (x - w // 2, y - h // 2, z - d // 2), (0, 0, 1), d)
        draw_line_3d(raw_img, (x + w // 2, y - h // 2, z - d // 2), (0, 0, 1), d)
        draw_line_3d(raw_img, (x - w // 2, y + h // 2, z - d // 2), (0, 0, 1), d)
        draw_line_3d(raw_img, (x + w // 2, y + h // 2, z - d // 2), (0, 0, 1), d)


def draw_bbox_3d_(raw_img, bboxes):
    for bbox in bboxes:
        #print(bbox)
        x1, x2, y1, y2, z1, z2, _ = bbox
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        z1 = int(z1)
        z2 = int(z2)
        '''
        draw_line_3d(raw_img, (x1, y1, z1), (1, 0, 0), x2 - x1)
        draw_line_3d(raw_img, (x1, y1, z2), (1, 0, 0), x2 - x1)
        draw_line_3d(raw_img, (x1, y2, z1), (1, 0, 0), x2 - x1)
        draw_line_3d(raw_img, (x1, y2, z2), (1, 0, 0), x2 - x1)

        draw_line_3d(raw_img, (x1, y1, z1), (0, 1, 0), y2 - y1)
        draw_line_3d(raw_img, (x1, y1, z2), (0, 1, 0), y2 - y1)
        draw_line_3d(raw_img, (x2, y1, z1), (0, 1, 0), y2 - y1)
        draw_line_3d(raw_img, (x2, y1, z2), (0, 1, 0), y2 - y1)

        draw_line_3d(raw_img, (x1, y1, z1), (0, 0, 1), z2 - z1)
        draw_line_3d(raw_img, (x2, y1, z1), (0, 0, 1), z2 - z1)
        draw_line_3d(raw_img, (x1, y2, z1), (0, 0, 1), z2 - z1)
        draw_line_3d(raw_img, (x2, y2, z1), (0, 0, 1), z2 - z1)
        '''

        for i in range(z1, z2):
            draw_line_3d(raw_img, (x1, y1, i), (1, 0, 0), x2 - x1)
            draw_line_3d(raw_img, (x1, y2, i), (1, 0, 0), x2 - x1)
            draw_line_3d(raw_img, (x1, y1, i), (0, 1, 0), y2 - y1)
            draw_line_3d(raw_img, (x2, y1, i), (0, 1, 0), y2 - y1)

    return raw_img

def draw_bbox_3d(raw_img, num_class, center_pts, outputs_reg, crop, tooth_list, output_path):
    img_size = raw_img.shape[0]
    crop_size = crop[1] - crop[0]
    for c in range(num_class):
        cur_img = raw_img.copy()
        x = center_pts[c][0]
        y = center_pts[c][1]
        z = center_pts[c][2]
        x = int(img_size * x / crop_size)
        y = int(img_size * y / crop_size)
        z = int(img_size * z / crop_size)

        w = int(outputs_reg[c][0][0] * img_size)
        h = int(outputs_reg[c][0][1] * img_size)
        d = int(outputs_reg[c][0][2] * img_size)
        w = max(w, h)
        h = max(w, h)

        #x1, x2, y1, y2, z1, z2, _ = bbox
        x1 = x - w // 2
        x2 = x + w // 2
        y1 = y - h // 2
        y2 = y + h // 2
        z1 = z - d // 2
        z2 = z + d // 2
        draw_line_3d(cur_img, (x1, y1, z1), (1, 0, 0), x2 - x1)
        draw_line_3d(cur_img, (x1, y1, z2), (1, 0, 0), x2 - x1)
        draw_line_3d(cur_img, (x1, y2, z1), (1, 0, 0), x2 - x1)
        draw_line_3d(cur_img, (x1, y2, z2), (1, 0, 0), x2 - x1)
        draw_line_3d(cur_img, (x1, y1, z1), (0, 1, 0), y2 - y1)
        draw_line_3d(cur_img, (x1, y1, z2), (0, 1, 0), y2 - y1)
        draw_line_3d(cur_img, (x2, y1, z1), (0, 1, 0), y2 - y1)
        draw_line_3d(cur_img, (x2, y1, z2), (0, 1, 0), y2 - y1)
        draw_line_3d(cur_img, (x1, y1, z1), (0, 0, 1), z2 - z1)
        draw_line_3d(cur_img, (x2, y1, z1), (0, 0, 1), z2 - z1)
        draw_line_3d(cur_img, (x1, y2, z1), (0, 0, 1), z2 - z1)
        draw_line_3d(cur_img, (x2, y2, z1), (0, 0, 1), z2 - z1)

        save_as_raw(output_path[:-4] + '_' + str(tooth_list[c]) + '.raw', cur_img)

    #return raw_img


def draw_seg_result(raw_img, seg_list):
    raw_img = np.uint8(raw_img * 0.5)
    tooth_num = 11
    while tooth_num < 49:
        if tooth_num % 10 == 9:
            tooth_num += 2
            continue

        pred_mask = seg_list[str(tooth_num)]
        raw_img[pred_mask == 1] = 255
        tooth_num += 1

    return raw_img


if __name__ == '__main__':
    raw_path = 'D:\\data\\osstem_tooth_CT\\rawdata\\TEST\\704_01\\ct'
    csv_path = 'D:\\data\\healthhub_tooth_CT\\experiments\\hg_final_gd_fix_thresh\\results\\toothnet_results_704_01.csv'
    out_path = 'D:\\data\\healthhub_tooth_CT\\experiments\\hg_final_gd_fix_thresh\\results\\'
    from utils.preprocess_img import dcm_to_np
    import pandas as pd
    crop = (0, 752, 0, 752, 0, 450)
    ct_array, _, _ = dcm_to_np(raw_path, crop, None, do_zoom=False)
    print(ct_array.shape)

    df = pd.read_csv(csv_path)
    bboxes = []
    for i, row in df.iterrows():
        bboxes.append([row['x1'], row['x2'], row['y1'], row['y2'], row['z1'], row['z2'], 0])

    ct_array = draw_bbox_3d_(ct_array, bboxes)
    #ct_array = ct_array.transpose(2,0,1)
    print(ct_array.shape)
    save_as_raw(out_path + '704_01_test_toothnet.raw', ct_array)
