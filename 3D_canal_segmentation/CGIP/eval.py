import os
import sys
import json
import cv2
import numpy as np
import math
import argparse


def load_gt(txt_path, orig_size):
    H, W = orig_size
    gt_list = {}
    gt_count = 0

    with open(txt_path, 'r') as gt_txt:
        content = gt_txt.readlines()
        for line in content:
            tooth_data = line.split(',')

            # Skip for empty line
            if len(tooth_data) < 2:
                continue

            tooth_num = int(tooth_data[0])

            # count only existing tooth
            if not is_missing_tooth(tooth_data):
                gt_count += 1

            # Calculate bbox center position
            x_min = y_min = 99999
            x_max = y_max = 0
            i = 3
            while i < 19:
                x = int(tooth_data[i]) + W // 2
                y = int(tooth_data[i+1]) + H // 2
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
                i += 2

            bbox_coords = (x_min, y_min, x_max, y_max)
            x_center = (x_min + x_max) // 2
            y_center = (y_min + y_max) // 2

            gt_list[tooth_num] = {'bbox_coords': bbox_coords,
                                  'center': (x_center, y_center),
                                  'area': get_area(bbox_coords),
                                  'is_missing': is_missing_tooth(tooth_data),
                                  'selected': False}

        return gt_list, gt_count


def load_det(result_path, orig_size, is_normalized):
    H, W = orig_size
    det_count = 0

    with open(result_path) as det_json:
        dets = json.load(det_json)

        for d in dets:
            tooth_center_initial = d['center_initial']
            tooth_center_final = d['center_final']
            bbox_w, bbox_h = d['bbox_wh']

            x1 = tooth_center_initial[0] - bbox_w / 2
            x2 = tooth_center_initial[0] + bbox_w / 2
            y1 = tooth_center_initial[1] - bbox_h / 2
            y2 = tooth_center_initial[1] + bbox_h / 2
            if is_normalized:
                x1 *= W
                x2 *= W
                y1 *= H
                y2 *= H

            d['bbox_coords_initial'] = (x1, y1, x2, y2)
            d['area'] = get_area(d['bbox_coords_initial'])

            if tooth_center_final is None:
                d['bbox_coords_final'] = None
            else:
                x1 = tooth_center_final[0] - bbox_w / 2
                x2 = tooth_center_final[0] + bbox_w / 2
                y1 = tooth_center_final[1] - bbox_h / 2
                y2 = tooth_center_final[1] + bbox_h / 2
                if is_normalized:
                    x1 *= W
                    x2 *= W
                    y1 *= H
                    y2 *= H

                d['bbox_coords_final'] = (x1, y1, x2, y2)

            d['selected'] = False
            det_count += 1

        return dets, det_count


def is_missing_tooth(tooth_data):
    s = str(tooth_data[1])
    if 'F' in s or 'f' in s:
        return True

    return False


def get_squared_error(bbox1, bbox2):
    """
    :param bbox1: aabb coord (x1, y1, x2, y2)
    :param bbox2: aabb coord (x3, y3, x4, y4)
    :return: squared error of bbox centers
    """
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    return ((x1 + x2 - x3 - x4)**2 + (y1 + y2 - y3 - y4)**2) / 4


def get_squared_error_center(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2
    return ((x1 - x2)**2 + (y1 - y2)**2) / 4


def rescale_center(coord, orig_size, target_size=(512,768)):
    """
    rescale coord from orig size to target size
    """
    x, y = coord
    x *= target_size[0] / orig_size[0]
    y *= target_size[0] / orig_size[0]
    return (x, y)


def get_intersection(bbox1, bbox2):
    """
    :param bbox1: aabb coord (x1, y1, x2, y2)
    :param bbox2: aabb coord (x3, y3, x4, y4)
    :return: aabb coord of intersection bbox
    """
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    return (max(x1, x3), max(y1, y3), min(x2, x4), min(y2, y4))


def get_area(bbox):
    """
    :param bbox: aabb coord (x1, y1, x2, y2)
    :return: area of bbox
    """
    x1, y1, x2, y2 = bbox
    if x1 > x2 or y1 > y2:
        return 0

    return (x2 - x1) * (y2 - y1)


def calculate_AP(results_list):
    """
    calculate AP(Average Precision) with 11-point interpolation
    Ref: https://github.com/rafaelpadilla/Object-Detection-Metrics
    """
    AP_list = []

    for i in range(19):
        target_recall = 100
        max_precision = 0
        precision_sum = 0
        for d in reversed(results_list):
            while d['recall'][i] * 100 < target_recall and target_recall >= 0:
                precision_sum += max_precision
                target_recall -= 10

            max_precision = max(max_precision, d['precision'][i])

        while target_recall >= 0:
            precision_sum += max_precision
            target_recall -= 10

        AP_list.append(precision_sum / 11)

    return AP_list


def evaluate(configs, target='initial'):
    """
    calculate AP, mIOU, MSE of det bboxes
    """
    print('\n-------------------------------------------')
    print('Evaluating detection bboxes by mAP and mIOU for', target, 'points')
    print('-------------------------------------------')
    # TODO: rule out MSE from this function
    results_list = []
    total_gt_count = 0
    total_det_count_for_mse = 0

    # for each result json files
    for f in os.listdir(configs.result_dir):
        if f[-4:] != 'json':
            continue

        # load img
        img_id = f.split('_')[0]
        img = cv2.imread(os.path.join(configs.data_dir, img_id + '.jpg'), cv2.IMREAD_COLOR)
        H, W, _ = img.shape

        # load gt txt file
        gt_list, gt_count = load_gt(os.path.join(configs.data_dir, img_id + '.txt'), (H, W))
        total_gt_count += gt_count

        # load detection result file
        dets, det_count = load_det(os.path.join(configs.result_dir, f), (H, W), configs.is_normalized)

        # if no final points, do nothing
        if target == 'final':
            if dets[0]['bbox_coords_final'] is None:
                print('no final points')
                return

        # if numbered, ignore detections of tooth number, which gt is missing tooth
        if configs.numbered:
            for d in dets:
                if gt_list[d['tooth_num']]['is_missing']:
                    d['selected'] = True
                    det_count -= 1

        while True:
            # until run out of gt or det
            if gt_count == 0 or det_count == 0:
                break

            max_IOU = 0
            max_idx_gt = None
            max_idx_det = None

            # find max IOU pair of det & gt
            for d_idx, d in enumerate(dets):
                if d['selected']:
                    continue

                for gt_idx, gt in gt_list.items():
                    if gt['is_missing'] or gt['selected']:
                        continue

                    intersection = get_intersection(gt['bbox_coords'], d['bbox_coords_' + target])
                    area_intersection = get_area(intersection)
                    IOU = area_intersection / (gt['area'] + d['area'] - area_intersection)
                    if max_IOU < IOU:
                        max_IOU = IOU
                        max_idx_gt = gt_idx
                        max_idx_det = d_idx

            # if no more pair, break
            if max_idx_gt is None or max_idx_det is None:
                break

            gt_list[max_idx_gt]['selected'] = True
            dets[max_idx_det]['selected'] = True
            gt_count -= 1
            det_count -= 1
            total_det_count_for_mse += 1

            # add to pair list
            this_pair = {}
            this_pair['confidence'] = 1 if configs.numbered else dets[max_idx_det]['score']
            this_pair['IOU'] = max_IOU
            this_pair['squared_error'] = get_squared_error(gt_list[max_idx_gt]['bbox_coords'], dets[max_idx_det]['bbox_coords_' + target])
            results_list.append(this_pair)

        # add leftover dets
        for d_idx, d in enumerate(dets):
            if d['selected']:
                continue

            d['selected'] = True
            this_pair = {}
            this_pair['confidence'] = 1 if configs.numbered else d['score']
            this_pair['IOU'] = 0
            this_pair['squared_error'] = None
            results_list.append(this_pair)

    accumulated_TP_list = [0 for _ in range(19)] # iou thresh: 5, 10, 15, ... 90, 95
    mIOU = 0
    MSE = 0

    # sort detections by confidence and IOU
    results_list.sort(key=lambda d: (d['confidence'], d['IOU']), reverse=True)

    # add precision & recall for all detections
    for idx, d in enumerate(results_list):
        mIOU += d['IOU']
        MSE += d['squared_error'] if d['squared_error'] is not None else 0
        d['precision'] = []
        d['recall'] = []

        for i in range(19):
            thresh = (i+1) * 0.05

            # set accumulated TP
            if d['IOU'] >= thresh:
                accumulated_TP_list[i] += 1

            # set precision
            d['precision'].append(accumulated_TP_list[i] / (idx + 1))
            d['recall'].append(accumulated_TP_list[i] / total_gt_count)

    # calculate mIOU and MSE
    mIOU /= len(results_list)
    MSE /= total_det_count_for_mse

    # calculate AP with 11-point interpolation
    AP_list = calculate_AP(results_list)
    mAP = sum(AP_list) / 19

    # print results
    print(AP_list)
    print('mAP: {:0.5f}, mIOU: {:0.5f}, MSE: {:0.5f}' \
        .format(mAP, mIOU, MSE))


def evaluate_centers(configs, target='initial'):
    """
    calculate only MSE of center points(32 class)
    rescaled to (512,768)
    """
    print('\n----------------------------------')
    print('Evaluating detection points by MSE for', target, 'points')
    print('----------------------------------')
    mse_list = [0 for _ in range(32)]
    gt_count = [0 for _ in range(32)]
    mse_total = 0
    total_count = 0

    # for each result json files
    for f in os.listdir(configs.result_dir):
        if f[-4:] != 'json':
            continue

        # load img
        img_id = f.split('_')[0]
        img = cv2.imread(os.path.join(configs.data_dir, img_id + '.jpg'), cv2.IMREAD_COLOR)
        H, W, _ = img.shape

        # load gt txt file
        gt_list, _ = load_gt(os.path.join(configs.data_dir, img_id + '.txt'), (H, W))

        # load detection result file
        dets, _ = load_det(os.path.join(configs.result_dir, f), (H, W), configs.is_normalized)

        # if no final points, do nothing
        if target == 'final':
            if dets[0]['bbox_coords_final'] is None:
                print('no final points')
                return

        for d in dets:
            tooth_num = int(d['tooth_num'])
            this_gt = gt_list[tooth_num]
            if this_gt['is_missing']:
                continue

            x_center, y_center = d['center_' + target]
            if configs.is_normalized:
                x_center *= W
                y_center *= H

            x_center, y_center = rescale_center((x_center, y_center), (H, W))
            gt_center_rescaled = rescale_center(this_gt['center'], (H, W))

            tooth_cls = (tooth_num // 10) * 8 + tooth_num % 10 - 9
            gt_count[tooth_cls] += 1
            total_count += 1

            squared_error = get_squared_error_center((x_center, y_center), gt_center_rescaled)
            mse_list[tooth_cls] += squared_error
            mse_total += squared_error

    # print final results
    mse_total /= total_count
    for i in range(32):
        tooth_num = (i // 8) * 10 + i % 8 + 11
        mse_list[i] /= gt_count[i]
        print('tooth_num: {}, MSE: {:0.5f}'.format(tooth_num, mse_list[i]))

    print('total MSE: {:0.5f}'.format(mse_total))


def get_confusion_matrix(configs, IOU_thresh=0.5, target='initial'):
    """
    calculate confusion matrix for 32 class detection
    """
    print('\n----------------------------------')
    print('Building Confusion Matrix for', target, 'points')
    print('----------------------------------')

    confusion_matrix = [[0 for i in range(33)] for j in range(33)] # 33 = 32 + background

    # for each result json files
    for f in os.listdir(configs.result_dir):
        if f[-4:] != 'json':
            continue

        # load img
        img_id = f.split('_')[0]
        img = cv2.imread(os.path.join(configs.data_dir, img_id + '.jpg'), cv2.IMREAD_COLOR)
        H, W, _ = img.shape

        # load gt txt file
        gt_list, _ = load_gt(os.path.join(configs.data_dir, img_id + '.txt'), (H, W))

        # load detection result file
        dets, _ = load_det(os.path.join(configs.result_dir, f), (H, W), configs.is_normalized)

        # if no final points, do nothing
        if target == 'final':
            if dets[0]['bbox_coords_final'] is None:
                print('no final points')
                return

        # calculate confusion matrix
        # reference: https://towardsdatascience.com/confusion-matrix-in-object-detection-with-tensorflow-b9640a927285
        for gt_tooth_num, gt in gt_list.items():
            if gt['is_missing']:
                continue

            gt_tooth_cls = (gt_tooth_num // 10) * 8 + gt_tooth_num % 10 - 9

            # find det which is max IOU
            max_IOU = 0
            max_IOU_idx = None
            for i, d in enumerate(dets):
                if d['selected']:
                    continue

                intersection = get_intersection(gt['bbox_coords'], d['bbox_coords_' + target])
                area_intersection = get_area(intersection)
                IOU = area_intersection / (gt['area'] + d['area'] - area_intersection)
                if max_IOU < IOU:
                    max_IOU = IOU
                    max_IOU_idx = i

            if max_IOU < 0.5:
                # if max_IOU is lower than thresh, add to last column(background)
                confusion_matrix[gt_tooth_cls][32] += 1
            else:
                dets[max_IOU_idx]['selected'] = True
                det_tooth_num = dets[max_IOU_idx]['tooth_num']
                det_tooth_cls = (det_tooth_num // 10) * 8 + det_tooth_num % 10 - 9
                confusion_matrix[gt_tooth_cls][det_tooth_cls] += 1

        # add remaining dets to last row
        for d in dets:
            if d['selected']:
                continue

            det_tooth_num = d['tooth_num']
            if gt_list[det_tooth_num]['is_missing']:
                continue

            det_tooth_cls = (det_tooth_num // 10) * 8 + det_tooth_num % 10 - 9
            confusion_matrix[32][det_tooth_cls] += 1

    # print confusion matrix result
    for row in confusion_matrix:
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='E:\\Data\\Teeth-Pano\\test')
    parser.add_argument('--result_dir', default='E:\\Data\\Teeth-Pano\\results\\')
    parser.add_argument('--is_normalized', default=False)
    parser.add_argument('--numbered', default=True)
    parser.add_argument('--only_center', default=False)
    parser.add_argument('--confusion_matrix', default=True)
    configs = parser.parse_args()

    if not configs.only_center:
        evaluate(configs, target='initial')
        evaluate(configs, target='final')

    evaluate_centers(configs, target='initial')
    evaluate_centers(configs, target='final')

    if configs.confusion_matrix:
        get_confusion_matrix(configs, target='initial')
        get_confusion_matrix(configs, target='final')
