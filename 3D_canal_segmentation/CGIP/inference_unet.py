import os
import argparse
from argparse import RawTextHelpFormatter
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.MaskDataset import MaskDataset
from networks.unet_3d import UNet_3D
import numpy as np
from scipy.ndimage.interpolation import zoom
from utils.visualize import save_as_raw

device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == "__main__":
    # Parsing argument
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

    parser.add_argument('--prj_name', type=str, default='', help='project name')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size (default: 2)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')

    # Training options
    parser.add_argument('--crop_info', type=int, nargs=6, default=(96, 416, 0, 320, -320, 0),
                        help='model input image crop info size(x, x, y, y, z, z) (default: (96, 416, 0, 320, -320, 0))')
    parser.add_argument('--resume', type=int, default=0, help='resume from checkpoint')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')

    args.prj_name = 'unet_mini_flip'
    args.dataset_name = 'cropped_mini'
    args.root_path = 'D:\\data\\healthhub_tooth_CT'
    args.data_path = os.path.join(args.root_path, 'gendata', args.dataset_name)
    args.meta_path = os.path.join(args.root_path, 'metadata', args.dataset_name + '.csv')
    args.model_save_path = os.path.join(args.root_path, 'experiments', args.prj_name)
    args.log_path = os.path.join(args.model_save_path, 'logs')
    if not os.path.exists(args.model_save_path):
        os.mkdir(args.model_save_path)
    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)

    args.include_wisdom = True
    num_tooth_type = 8 if args.include_wisdom else 7
    args.num_class = num_tooth_type * 4
    tooth_list = []
    for i in range(args.num_class):
        tooth_num = (i // num_tooth_type) * 10 + i % num_tooth_type + 11
        tooth_list.append(tooth_num)

    # Model
    net = UNet_3D(n_class=1)

    # load checkpoint
    checkpoint_epoch = 100
    checkpoint = torch.load(os.path.join(args.model_save_path, 'ckpt-' + str(checkpoint_epoch) + '.pth'))
    net.load_state_dict(checkpoint['net'], strict=True)
    net = net.to(device)
    net.eval()
    print("Model Done")

    # Data
    test_dataset = MaskDataset(
        mode='val',
        img_path=os.path.join(args.data_path, 'val', 'image_big'),
        gt_path=os.path.join(args.data_path, 'val', 'gt_dist_map'),
        meta_path=args.meta_path,
        crop=args.crop_info,
        tooth_list=tooth_list
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print("Dataset Done", args.num_class)
    crop_size = (args.crop_info[1] - args.crop_info[0],
                 args.crop_info[1] - args.crop_info[0],
                 args.crop_info[1] - args.crop_info[0])

    count = 0
    out_num = 10
    output_path = os.path.join(args.root_path, 'experiments', args.prj_name, 'results')
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if count == out_num:
                break

            inputs = inputs.to(device)
            targets = targets.numpy().squeeze()
            output = net(inputs).squeeze() # output: (128, 64, 64)
            output = output.cpu().numpy()
            print(np.min(output), np.max(output))

            #output[output < 0] = 0
            #output *= 8
            #output = np.uint8(output)

            img_name = test_dataset.get_img_filename(batch_idx)
            this_tooth_num = test_dataset.get_tooth_num(batch_idx)
            this_bbox = test_dataset.get_bbox(batch_idx)

            # Load img and crop
            img = np.fromfile(os.path.join(os.path.join(args.data_path, 'val', 'image_big'), img_name + '.raw'), dtype=np.uint8).reshape(crop_size)
            img = test_dataset.get_cropped_img(img, this_bbox) # (128, 64, 64)
            if int(this_tooth_num) < 30:
                img = np.flip(img, 0)

            img = img // 2
            img[output > 2] = 255

            save_as_raw(os.path.join(output_path, img_name + '_' + str(this_tooth_num) + '.raw'), img)
            count += 1
