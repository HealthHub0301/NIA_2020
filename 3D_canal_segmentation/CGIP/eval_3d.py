import os
import pandas as pd
import numpy as np


def get_intersection_3d(bbox1, bbox2):
    """
    :param bbox1: aabb coord (x1, x2, y1, y2, z1, z2)
    :param bbox2: aabb coord (x3, x4, y3, y4, z3, z4)
    :return: aabb coord of intersection bbox
    """
    x1, x2, y1, y2, z1, z2 = bbox1
    x3, x4, y3, y4, z3, z4 = bbox2
    return (max(x1, x3), min(x2, x4), max(y1, y3), min(y2, y4), max(z1, z3), min(z2, z4))


def get_area_3d(bbox):
    """
    :param bbox: aabb coord (x1, x2, y1, y2, z1, z2)
    :return: area of bbox
    """
    x1, x2, y1, y2, z1, z2 = bbox
    if x1 > x2 or y1 > y2 or z1 > z2:
        return 0

    return (x2 - x1) * (y2 - y1) * (z2 - z1)


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


def calculate_OIR(mask, crop, bbox, original_size=512):
    mask_area = 0
    included_area = 0
    crop_size = crop[1] - crop[0]
    for k in range(crop_size):
        for j in range(crop_size):
            for i in range(crop_size):
                val = mask[k, j, i]
                if val > 0:
                    mask_area += 1
                    x = i + crop[0]
                    y = j + crop[2]
                    z = k + crop[4] + original_size
                    if (bbox[0] <= x and x <= bbox[1] and
                        bbox[2] <= y and y <= bbox[3] and
                        bbox[4] <= z and z <= bbox[5]):
                        included_area += 1

    if mask_area == 0:
        return -1

    return included_area / mask_area

def evaluate(pred_df, gt_df, mask_path, tooth_list, do_OIR=False):
    results_list = []
    total_gt_count = 0
    crop = (96, 416, 0, 320, -320, 0) # x, x, y, y, z, z
    crop_width = crop[1] - crop[0]

    for i, pred_row in pred_df.iterrows():

        img_name = pred_row['name']
        #print(img_name)
        target_row = None
        for j, gt_row in gt_df.iterrows():
            if gt_row['name'] == img_name:
                target_row = gt_row
                break

        if target_row is None:
            print('image not found')
            return

        z_offset = 512 - int(eval(target_row['size'])[0])

        for t in tooth_list:
            t = str(t)
            det = {}

            if eval(target_row[t]) is None:
                #det['IOU'] = 0
                #results_list.append(det)
                continue

            total_gt_count += 1
            margin_1 = 15
            margin_2 = 5

            target_x_min, target_x_max, target_y_min, target_y_max, target_z_min, target_z_max\
                = eval(target_row[t])
            target_x_center = (target_x_min + target_x_max) / 2
            target_y_center = (target_y_min + target_y_max) / 2
            target_z_center = (target_z_min + target_z_max) / 2
            target_bbox = (
                target_x_min - margin_1,
                target_x_max + margin_1,
                target_y_min - margin_1,
                target_y_max + margin_1,
                target_z_min - margin_1 + z_offset,
                target_z_max + margin_1 + z_offset
            )

            pred_x_min, pred_x_max, pred_y_min, pred_y_max, pred_z_min, pred_z_max, _\
                = eval(pred_row[t])
            pred_bbox = (
                pred_x_min - margin_2,
                pred_x_max + margin_2,
                pred_y_min - margin_2,
                pred_y_max + margin_2,
                pred_z_min - 2 * margin_2,
                pred_z_max + 2 * margin_2
            )
            intersection = get_intersection_3d(pred_bbox, target_bbox)
            area_intersection = get_area_3d(intersection)
            area_gt = get_area_3d(target_bbox)
            area_pred = get_area_3d(pred_bbox)

            area_intersection = area_intersection ** (2 / 3)
            area_gt = area_gt ** (2 / 3)
            area_pred = area_pred ** (2 / 3)
            det['IOU'] = area_intersection / (area_pred + area_gt - area_intersection)
            det['OIR'] = 0

            if do_OIR:
                mask = np.fromfile(os.path.join(mask_path, img_name + '_' + t + '.raw'),
                                   dtype=np.uint8).reshape((crop_width, crop_width, crop_width))
                det['OIR'] = calculate_OIR(mask, crop, pred_bbox)
                print(t, det['OIR'])

            results_list.append(det)
            #print(t, pred_bbox, target_bbox, area_pred, area_gt, area_intersection, IOU)

    accumulated_TP_list = [0 for _ in range(19)]  # iou thresh: 5, 10, 15, ... 90, 95
    mIOU = 0
    mOIR = 0

    # sort detections by confidence and IOU
    results_list.sort(key=lambda d: d['IOU'], reverse=True)

    # add precision & recall for all detections
    for idx, det in enumerate(results_list):
        mIOU += det['IOU']
        mOIR += det['OIR']
        det['precision'] = []
        det['recall'] = []

        for i in range(19):
            thresh = (i + 1) * 0.05

            # set accumulated TP
            if det['IOU'] >= thresh:
                accumulated_TP_list[i] += 1

            # set precision
            det['precision'].append(accumulated_TP_list[i] / (idx + 1))
            det['recall'].append(accumulated_TP_list[i] / total_gt_count)

    # calculate mIOU and MSE
    mIOU /= len(results_list)
    #mOIR /= len(results_list)
    mOIR /= 5

    # calculate AP with 11-point interpolation
    AP_list = calculate_AP(results_list)
    mAP = sum(AP_list) / 19

    # print results
    print(AP_list)
    print('mAP: {:0.5f}, AP50: {:0.5f}, mIOU: {:0.5f}, mOIR: {:0.5f}' \
          .format(mAP, AP_list[9], mIOU, mOIR))
    print('precision: {:0.5f}, recall: {:0.5f}'\
          .format(accumulated_TP_list[9] / len(results_list), accumulated_TP_list[9] / total_gt_count))


def evaluate_identification(pred_df, gt_df, tooth_list):
    results_list = []
    total_gt_count = 0
    total_tp = 0
    total_tpfp = 0
    crop = (96, 416, 0, 320, -320, 0) # x, x, y, y, z, z
    crop_width = crop[1] - crop[0]

    cm = [[0 for i in range(num_class + 1)] for j in range(num_class + 1)]

    for i, pred_row in pred_df.iterrows():
        img_name = pred_row['name']
        #print(img_name)
        target_row = None
        for j, gt_row in gt_df.iterrows():
            if gt_row['name'] == img_name:
                target_row = gt_row
                break

        if target_row is None:
            print('image not found')
            return

        z_offset = 512 - int(eval(target_row['size'])[0])
        margin_1 = 0
        margin_2 = 0
        hm_thresh = 0.07

        for t_target in range(num_class):
            if eval(target_row[str(tooth_list[t_target])]) is None:
                hm_max = eval(pred_row[str(tooth_list[t_target])])[6]
                if hm_max > hm_thresh:
                    cm[num_class][t_target] += 1

                continue

            total_gt_count += 1
            target_x_min, target_x_max, target_y_min, target_y_max, target_z_min, target_z_max \
                = eval(target_row[str(tooth_list[t_target])])
            #target_x_center = (target_x_min + target_x_max) / 2
            #target_y_center = (target_y_min + target_y_max) / 2
            #target_z_center = (target_z_min + target_z_max) / 2
            target_bbox = (
                target_x_min - margin_1,
                target_x_max + margin_1,
                target_y_min - margin_1,
                target_y_max + margin_1,
                target_z_min - margin_1 + z_offset,
                target_z_max + margin_1 + z_offset
            )

            max_intersection_area = 0
            max_pred_t = 0
            hm_val = 0
            for t_pred in range(num_class):
                pred_x_min, pred_x_max, pred_y_min, pred_y_max, pred_z_min, pred_z_max, hm_max \
                    = eval(pred_row[str(tooth_list[t_pred])])
                pred_bbox = (
                    pred_x_min - margin_2,
                    pred_x_max + margin_2,
                    pred_y_min - margin_2,
                    pred_y_max + margin_2,
                    pred_z_min - 2 * margin_2,
                    pred_z_max + 2 * margin_2
                )

                intersection = get_intersection_3d(pred_bbox, target_bbox)
                area_intersection = get_area_3d(intersection)

                if max_intersection_area < area_intersection:
                    max_intersection_area = area_intersection
                    max_pred_t = t_pred
                    hm_val = hm_max

            if hm_val > hm_thresh:
                cm[t_target][max_pred_t] += 1
                total_tpfp += 1
                if max_pred_t == t_target:
                    total_tp += 1
            else:
                cm[t_target][num_class] += 1


    '''
    for i, pred_row in pred_df.iterrows():
        img_name = pred_row['name']
        print(img_name)
        target_row = None
        for j, gt_row in gt_df.iterrows():
            if gt_row['name'] == img_name:
                target_row = gt_row
                break

        if target_row is None:
            print('image not found')
            return

        z_offset = 512 - int(eval(target_row['size'])[0])
        margin_1 = 13
        margin_2 = 5

        for t in tooth_list:
            if eval(target_row[str(t)]) is not None:
                total_gt_count += 1

            hm_max = eval(pred_row[str(t)])[6]
            if hm_max >= 0.15:
                total_tpfp += 1

        for t in tooth_list:
            if eval(target_row[str(t)]) is None:
                continue

            pred_x_min, pred_x_max, pred_y_min, pred_y_max, pred_z_min, pred_z_max, hm_max \
                = eval(pred_row[str(t)])

            if hm_max < 0.1:
                continue

            pred_bbox = (
                pred_x_min - margin_2,
                pred_x_max + margin_2,
                pred_y_min - margin_2,
                pred_y_max + margin_2,
                pred_z_min - 2 * margin_2,
                pred_z_max + 2 * margin_2
            )

            target_x_min, target_x_max, target_y_min, target_y_max, target_z_min, target_z_max \
                = eval(target_row[str(t)])
            target_bbox = (
                target_x_min - margin_1,
                target_x_max + margin_1,
                target_y_min - margin_1,
                target_y_max + margin_1,
                target_z_min - margin_1 + z_offset,
                target_z_max + margin_1 + z_offset
            )

            intersection = get_intersection_3d(pred_bbox, target_bbox)
            area_intersection = get_area_3d(intersection)
            area_gt = get_area_3d(target_bbox)
            area_pred = get_area_3d(pred_bbox)

            area_intersection = area_intersection ** (2 / 3)
            area_gt = area_gt ** (2 / 3)
            area_pred = area_pred ** (2 / 3)
            IOU = area_intersection / (area_pred + area_gt - area_intersection)
            if IOU >= 0.5:
                total_tp += 1
    '''

    #print(total_tp, total_tpfp, total_gt_count)
    print('precision: {:0.5f}, recall: {:0.5f}'\
          .format(total_tp / total_tpfp, total_tp / total_gt_count))
    for cm_r in cm:
        print(cm_r)



if __name__ == "__main__":
    root_path = 'D:\\data\\healthhub_tooth_CT'
    #prj_name = 'hg_32_focal'
    #prj_name = 'hg_centernet_32'
    #prj_name = 'hg_final_gd_fix_thresh'

    #prj_names = ['hg_32_focal', 'hg_centernet_32', 'hg_final_gd_fix_thresh']
    prj_names = ['hg_final_gd_fix_thresh']

    dataset_name = 'cropped_100'

    meta_path = os.path.join(root_path, 'metadata', dataset_name + '.csv')
    mask_path = os.path.join(root_path, 'gendata', dataset_name, 'val', 'gt_mask')

    num_tooth_type = 7
    num_class = num_tooth_type * 4
    tooth_list = []
    for i in range(num_class):
        tooth_num = (i // num_tooth_type) * 10 + i % num_tooth_type + 11
        tooth_list.append(tooth_num)

    gt_df = pd.read_csv(meta_path)

    for prj_name in prj_names:
        output_path = os.path.join(root_path, 'experiments', prj_name, 'results', 'results.csv')
        pred_df = pd.read_csv(output_path)
        evaluate(pred_df, gt_df, mask_path, tooth_list, do_OIR=False)
        #evaluate_identification(pred_df, gt_df, tooth_list)
