import os
import argparse
from argparse import RawTextHelpFormatter
import json
import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader

from utils.TeethDataset import TeethDataset
from networks.ResNet import ResNet18
from networks.Pano import Pano

from utils.preprocess_img import restore_coord
from utils.visualize import visualize_compare, visualize, save_cropped_teeth, visualize_pts

def test(model, data_loader, device, args):
    assert('target' in args)

    model.eval()

    with torch.no_grad():
        for idx, (inputs, gt_pts, gt_boxes) in enumerate(data_loader):
            bbox = []
            inputs = inputs.to(device)

            out_pts, out_bbox, out_offset = model(inputs)

            out_origin_pts = out_pts.clone().detach()

            # Third output
            if args.target == "offset":
                out_pts = out_pts + out_offset

            elif args.target == "center":
                target_origins = out_pts.clone().detach()
                patch_size = list(model.module.get_patch_size())

                target_origins[:, 0::2] -= patch_size[0] / 2
                target_origins[:, 1::2] -= patch_size[1] / 2

                out_pts = target_origins + out_offset

            elif args.target == "none":
                pass

            else:
                raise ValueError('Not allowed loss_target option')

            min_bbox_pts = out_pts - out_bbox / 2
            max_bbox_pts = out_pts + out_bbox / 2

            for batch_idx in range(inputs.shape[0]):
                for i in range(min_bbox_pts.shape[1]//2):
                    bbox.append(((min_bbox_pts[batch_idx, 2*i].item(), min_bbox_pts[batch_idx, 2*i+1].item()),
                                (max_bbox_pts[batch_idx, 2*i].item(), max_bbox_pts[batch_idx, 2*i+1].item())))

            img_path = data_loader.dataset.get_img_filename(idx)
            visualize_pts(img_path, args.dst, size=inputs.shape[-2:], org_pts=out_pts[0].clone().detach())

            # Save GT
            if args.save_gt is True:
                visualize(img_path, args.dst, gt_pts[0].clone().detach(), gt_boxes.tolist()[0],
                          gt_boxes.tolist()[0],
                          size=inputs.shape[-2:], is_gt=True)

            # Save visualized result
            visualize(img_path, args.dst, out_pts[0].clone().detach(), bbox.copy(),
                      gt_boxes.tolist()[0],
                      size=inputs.shape[-2:])

            # Save cropped teeth
            save_cropped_teeth(img_path, args.dst, bbox.copy(), size=inputs.shape[-2:])

            # For visualizing to compare results
            visualize_compare(img_path, args.dst, gt_pts[0].clone().detach(), gt_boxes.tolist()[0],
                              out_origin_pts[0].clone().detach(),
                              out_pts[0].clone().detach(),
                              bbox.copy(), size=inputs.shape[-2:], is_normalized=False)

            # Save results
            img_id = os.path.split(img_path)[1][:-4]
            if args.target =="none":
                save_json(img_path, os.path.join(args.dst, img_id + "_res.json"),
                          out_pts[0].clone().detach(),
                          None,
                          bbox.copy(), size=inputs.shape[-2:])
            else:
                save_json(img_path, os.path.join(args.dst, img_id + "_res.json"),
                          out_origin_pts[0].clone().detach(),
                          out_pts[0].clone().detach(), bbox.copy(), size=inputs.shape[-2:])

            # Progress bar
            data_cnt = len(data_loader.dataset)
            done_cnt = min((idx + 1) * data_loader.batch_size, data_cnt)
            rate = done_cnt / data_cnt
            bar = ('=' * int(rate * 32) + '>').ljust(32, '.')
            idx = str(done_cnt).rjust(len(str(data_cnt)), ' ')
            print('\rInference\t : {}/{}: [{}]'.format(
                idx,
                data_cnt,
                bar
            ), end='')


def save_json(img_src, dst, pts_initial, pts_final, bboxes, size):

    img = cv2.imread(img_src, cv2.IMREAD_COLOR)
    h, w, _ = img.shape

    with open(dst, "w") as json_file:
        teeth = []

        for i in range(32):
            tooth = {}

            tooth_num = (i // 8) * 10 + (i % 8) + 11
            tooth["tooth_num"] = tooth_num

            y_init = pts_initial[2 * i]
            x_init = pts_initial[2 * i + 1]

            y_init, x_init = restore_coord((y_init, x_init), size, (h, w))

            tooth["center_initial"] = [x_init, y_init]

            if pts_final is not None:
                y_final = pts_final[2 * i]
                x_final = pts_final[2 * i + 1]

                y_final, x_final = restore_coord((y_final, x_final), size, (h, w))

                tooth["center_final"] = [x_final, y_final]
            else:
                tooth["center_final"] = None

            min_point, max_point = bboxes[i]
            min_point = restore_coord(min_point, size, (h, w))
            max_point = restore_coord(max_point, size, (h, w))

            width = int(max_point[1] - min_point[1])
            height = int(max_point[0] - min_point[0])

            tooth["bbox_wh"] = [width, height]

            teeth.append(tooth)

        json.dump(teeth, json_file)


def main():
    # Parsing argument
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--model', required=True,
                        help='Model path')
    parser.add_argument('--data_path', required=True,
                        help='Directory path of test images')
    parser.add_argument('--dst', required=True,
                        help='Destination path for saving results')
    parser.add_argument('--save_gt', action='store_true', default=False, help='')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')

    # Load checkpoint
    checkpoint_path = args.model
    checkpoint = torch.load(checkpoint_path)

    args.image_size = checkpoint['image_size']
    args.patch_size = checkpoint['patch_size']
    args.target = checkpoint['loss_target']
    if 'backbone' in checkpoint:
        args.backbone = checkpoint['backbone']
    else:
        args.backbone = 'resnet'

    print(args)

    # Data
    testds = TeethDataset(path=args.data_path)
    test_loader = torch.utils.data.DataLoader(testds, batch_size=1, shuffle=False, num_workers=1)

    # Model
    net = Pano(image_size=args.image_size, patch_size=args.patch_size, backbone=args.backbone, target=args.target)
    net = torch.nn.DataParallel(net).to(device)
    net.load_state_dict(checkpoint['net'], strict=False)

    test(net, test_loader, device, args)

if __name__ == "__main__":
    main()
