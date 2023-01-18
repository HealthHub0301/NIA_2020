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

from utils.ToothCTDataset import ToothCTDataset
from utils.preprocess_img import get_meta_dict
from networks.ResNet import ResNet18
from networks.HourglassNet3D import hg_3d
from utils.loss import FocalLoss, get_bbox_reg_loss, _sigmoid
import pandas as pd
device = 'cuda' if torch.cuda.is_available() else 'cpu'
hm_thresh = nn.Threshold(0.25, 0)


# Training
def train(model, epoch, data_loader, hm_criterion, reg_criterion, optimizer, writer, device, args):
    model.train()

    print('\n==> epoch %d' % epoch)
    train_loss = 0.0
    train_hm_loss = 0.0
    train_gd_loss = 0.0
    train_reg_loss = 0.0
    num_inputs = 0
    outputs_hm = None
    outputs_reg = None

    for batch_idx, (inputs, hm_targets, reg_targets) in enumerate(data_loader):
        # zero the parameter gradients
        optimizer.zero_grad()

        inputs, hm_targets, reg_targets = inputs.to(device), hm_targets.to(device), reg_targets.to(device)
        if args.separate_head:
            outputs_hm, outputs_reg = model(inputs) # outputs_hm: (s, c, b, 1, d, h, w), outputs_reg: (c, b, num_reg)
        else:
            outputs_hm, outputs_reg = model(inputs)  # outputs_hm: (s, b, c, d, h, w), outputs_reg: (1, b, c * num_reg)

        num_stack = len(outputs_hm)
        num_type = args.num_class // 4
        hm_loss = 0
        gd_loss = 0

        if args.separate_head:
            for out_hm in outputs_hm:
                hm_sigmoid = []
                for c in range(args.num_class):
                    hm_sigmoid.append(torch.sigmoid(out_hm[c][:, 0]))

                for c in range(args.num_class):
                    hm_loss += hm_criterion(hm_sigmoid[c], hm_targets[:, c])
                    if args.weight[1] > 0 and c % num_type != 0:
                        gd_loss += torch.sum(torch.mul(hm_thresh(hm_sigmoid[c]), hm_thresh(hm_sigmoid[c - 1])))

                if args.weight[1] > 0:
                    gd_loss += torch.sum(torch.mul(hm_thresh(hm_sigmoid[0]), hm_thresh(hm_sigmoid[num_type])))
                    gd_loss += torch.sum(torch.mul(hm_thresh(hm_sigmoid[2 * num_type]), hm_thresh(hm_sigmoid[3 * num_type])))

            hm_loss /= num_stack * args.num_class
            gd_loss /= num_stack * ((num_type - 1) * 4 + 2)
            reg_loss = get_bbox_reg_loss(outputs_reg, reg_targets, reg_criterion) / args.num_class
        else:
            for out_hm in outputs_hm:
                hm = torch.sigmoid(out_hm)
                hm_loss += hm_criterion(hm, hm_targets)

            hm_loss /= num_stack * args.num_class
            reg_loss = reg_criterion(outputs_reg[0], reg_targets.view(args.batch_size, -1)) / args.num_class

        loss = args.weight[0] * hm_loss + args.weight[1] * gd_loss + args.weight[2] * reg_loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_hm_loss += hm_loss.item()
        if args.weight[1] > 0:
            train_gd_loss += gd_loss.item()
        train_reg_loss += reg_loss.item()
        num_inputs += args.batch_size

    train_loss /= num_inputs
    train_hm_loss /= num_inputs
    train_gd_loss /= num_inputs
    train_reg_loss /= num_inputs

    print("------Train------")
    print("Heatmap Loss(Focal): %f" % train_hm_loss)
    print("Gaussian Disentanglement Loss: %f" % train_gd_loss)
    print("Bbox Size Loss(MSE): %f" % train_reg_loss)
    print("Total Train Loss: %f"    % train_loss)
    print()
    writer.add_scalars("Loss/train", {
        "hm_loss" : train_hm_loss,
        "gd_loss" : train_gd_loss,
        "reg_loss" : train_reg_loss,
        "total_loss": train_loss,
    }, epoch)


# Validate
def evaluate(model, epoch, data_loader, hm_criterion, reg_criterion, writer, device, args):
    model.eval()

    val_loss = 0.0
    val_hm_loss = 0.0
    val_gd_loss = 0.0
    val_reg_loss = 0.0
    num_inputs = 0
    outputs_hm = None
    outputs_reg = None

    with torch.no_grad():
        for batch_idx, (inputs, hm_targets, reg_targets) in enumerate(data_loader):
            inputs, hm_targets, reg_targets = inputs.to(device), hm_targets.to(device), reg_targets.to(device)
            outputs_hm, outputs_reg = model(inputs) # outputs_hm: (c, b, 1, d, h, w), outputs_reg: (c, b, num_reg)

            num_stack = len(outputs_hm)
            num_type = args.num_class // 4
            hm_loss = 0
            gd_loss = 0
            if args.separate_head:
                for out_hm in outputs_hm:
                    hm_sigmoid = []
                    for c in range(args.num_class):
                        hm_sigmoid.append(torch.sigmoid(out_hm[c][:, 0]))

                    for c in range(args.num_class):
                        hm_loss += hm_criterion(hm_sigmoid[c], hm_targets[:, c])
                        if args.weight[1] > 0 and c % num_type != 0:
                            gd_loss += torch.sum(torch.mul(hm_thresh(hm_sigmoid[c]), hm_thresh(hm_sigmoid[c - 1])))

                    if args.weight[1] > 0:
                        gd_loss += torch.sum(torch.mul(hm_thresh(hm_sigmoid[0]), hm_thresh(hm_sigmoid[num_type])))
                        gd_loss += torch.sum(torch.mul(hm_thresh(hm_sigmoid[2 * num_type]), hm_thresh(hm_sigmoid[3 * num_type])))

                hm_loss /= num_stack * args.num_class
                gd_loss /= num_stack * ((num_type - 1) * 4 + 2)
                reg_loss = get_bbox_reg_loss(outputs_reg, reg_targets, reg_criterion) / args.num_class
            else:
                for out_hm in outputs_hm:
                    hm = torch.sigmoid(out_hm)
                    hm_loss += hm_criterion(hm, hm_targets)

                reg_loss = reg_criterion(outputs_reg[0], reg_targets.view(args.batch_size, -1)) / args.num_class

            loss = args.weight[0] * hm_loss + args.weight[1] * gd_loss + args.weight[2] * reg_loss

            val_loss += loss.item()
            val_hm_loss += hm_loss.item()
            if args.weight[1] > 0:
                val_gd_loss += gd_loss.item()
            val_reg_loss += reg_loss.item()
            num_inputs += args.batch_size

        val_loss /= num_inputs
        val_hm_loss /= num_inputs
        val_gd_loss /= num_inputs
        val_reg_loss /= num_inputs

        print("------Val------")
        print("Heatmap Loss(Focal): %f" % val_hm_loss)
        print("Gaussian Disentanglement Loss: %f" % val_gd_loss)
        print("Bbox Size Loss(MSE): %f" % val_reg_loss)
        print("Total Train Loss: %f"    % val_loss)
        print()
        writer.add_scalars("Loss/val", {
            "hm_loss" : val_hm_loss,
            "gd_loss" : val_gd_loss,
            "reg_loss" : val_reg_loss,
            "total_loss": val_loss,
        }, epoch)

    # Save checkpoint.
    if epoch % 10 == 0:
        if epoch > 0:
            state = {
                'epoch': epoch,
                'net': model.state_dict()
            }
            torch.save(state, os.path.join(args.model_save_path, "ckpt-" + str(epoch) + ".pth"))


if __name__ == "__main__":
    # Parsing argument
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

    parser.add_argument('--prj_name', type=str, default='', help='project name')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size (default: 12)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning_rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight_decay (default: 1e-5)')
    parser.add_argument('--max_epoch', type=int, default=300, help='maximum epoch (default: 300) ')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')

    # Training options
    parser.add_argument('--backbone', type=str, choices=['resnet', 'dla', 'hourglass'], default='hourglass',
                        help='')
    parser.add_argument('--input_size', type=int, nargs=3, default=(128, 128, 128),
                        help='model input image size (default: (128, 128, 128))')
    parser.add_argument('--output_size', type=int, nargs=3, default=(64, 64, 64),
                        help='model output heatmap size (default: (64, 64, 64))')
    parser.add_argument('--num_feats', type=int, default=32, help='number of feature map channels')
    parser.add_argument('--crop_info', type=int, nargs=6, default=(96, 416, 0, 320, -320, 0),
                        help='model input image crop info size(x, x, y, y, z, z) (default: (96, 416, 0, 320, -320, 0))')
    parser.add_argument('--weight', type=float, nargs=3, default=(1, 1, 0.1),
                        metavar=('WEIGHT1', 'WEIGHT2', 'WEIGHT3'),
                        help='weight of losses\n'
                             'heatmap_loss\n'
                             'gd_loss\n'
                             'bbox_loss')
    parser.add_argument('--include_wisdom', action='store_true', default=False,
                        help='Flag to include wisdom teeth')
    parser.add_argument('--do_reg', action='store_true', default=False,
                        help='Flag to perform bbox regression')
    parser.add_argument('--resume', type=int, default=0, help='resume from checkpoint')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')

    args.root_path = 'D:\\data\\healthhub_tooth_CT'
    args.prj_name = 'hg_transfer_centernet_gd_2'
    args.dataset_name = 'cropped_100'    

    args.data_path = os.path.join(args.root_path, 'gendata', args.dataset_name)
    args.meta_path = os.path.join(args.root_path, 'metadata', args.dataset_name + '.csv')
    args.model_save_path = os.path.join(args.root_path, 'experiments', args.prj_name)
    args.log_path = os.path.join(args.model_save_path, 'logs')
    if not os.path.exists(args.model_save_path):
        os.mkdir(args.model_save_path)
    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)

    args.num_feats = 28
    args.crop_info = (151, 601, 50, 500, 0, 450)
    args.do_reg = True
    args.include_wisdom = True
    args.separate_head = True
    num_tooth_type = 8 if args.include_wisdom else 7
    args.num_class = num_tooth_type * 4
    tooth_list = []
    for i in range(args.num_class):
        tooth_num = (i // num_tooth_type) * 10 + i % num_tooth_type + 11
        tooth_list.append(tooth_num)

    # Tensorboard
    writer = SummaryWriter(log_dir=args.log_path)

    # Model
    net = None
    if args.backbone == 'hourglass':
        '''
        hyperparameters:
            num_stacks: intermediate supervision을 적용하기 위한 stack의 개수
            num_blocks: hourglass의 각 단계의 redisdual block의 개수
            num_classes: 치아 종류의 개수
            num_feats: feature map의 채널의 개수
            in_channels: 인풋 이미지의 채널 수(흑백이라 1)
            hg_depth: hourglass의 깊이
            num_reg: regression 하고자 하는 수치의 개수(bbox의 depth height width 라서 3개)
            do_reg: bbox size regression을 할지말지
            separate_head: 각 치아 클래스 별 추가적인 complexity를 추가할지말지
        '''
        net = hg_3d(num_stacks=2, num_blocks=1, num_classes=args.num_class, num_feats=args.num_feats, in_channels=1, hg_depth=3, num_reg=3, do_reg=args.do_reg, separate_head=args.separate_head)

    # load checkpoint
    if args.resume > 0:
        checkpoint = torch.load(os.path.join(args.original_model_save_path, 'ckpt-' + str(args.resume) + '.pth'))
        net.load_state_dict(checkpoint['net'], strict=True)

    net = net.to(device)
    hm_criterion = FocalLoss().to(device)
    reg_criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print("Model Done")

    # Data
    meta_dict = get_meta_dict(args.meta_path, args.crop_info, tooth_list)
    train_dataset = ToothCTDataset(
        mode='train',
        img_path=os.path.join(args.data_path, 'train', 'image'),
        gt_path=os.path.join(args.data_path, 'train', 'gt_center_heatmap'),
        meta_dict=meta_dict,
        tooth_list=tooth_list,
        type='individual',
        target_size=args.input_size,
        do_reg=args.do_reg
    )
    val_dataset = ToothCTDataset(
        mode='train',
        img_path=os.path.join(args.data_path, 'val', 'image'),
        gt_path=os.path.join(args.data_path, 'val', 'gt_center_heatmap'),
        meta_dict=meta_dict,
        tooth_list=tooth_list,
        type='individual',
        target_size=args.input_size,
        do_reg=args.do_reg
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print("Dataset Done", args.num_class)

    # run
    for epoch in range(args.resume, args.max_epoch + 1):
        train(net, epoch, train_loader, hm_criterion, reg_criterion, optimizer, writer, device, args)
        evaluate(net, epoch, val_loader, hm_criterion, reg_criterion, writer, device, args)

    writer.close()
