import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import argparse
from argparse import RawTextHelpFormatter
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.TeethDataset import TeethDataset
from networks.ResNet import ResNet18
from networks.Pano import Pano
from utils.loss import get_SR_term


# Training
def train(model, epoch, data_loader, criterion, optimizer, writer, device, args):
    """
    Train model
    :param model: The torch module model to train which returns three outputs.
    :param epoch: The current number of epoch.
    :param data_loader: The data loader which loads input image and ground truths(pts, aabbs).
    :param criterion: The metric for loss calculation
    :param optimizer: The method of gradient descent.
    :param writer: Tensorboard writer to log model train losses.
    :param device: CPU or GPU
    :param loss_target: target third loss
    :return:
    """
    assert('target' in args)

    model.train()

    print('\n==> epoch %d' % epoch)
    train_center_loss = 0.0
    train_bbox_loss = 0.0
    train_offset_loss = 0.0
    train_sr_term_1 = 0.0
    train_sr_term_2 = 0.0
    train_loss = 0.0
    num_inputs = 0

    for batch_idx, (inputs, target_pts, target_boxes) in enumerate(data_loader):
        inputs, target_pts, target_boxes = \
            inputs.to(device), target_pts.to(device), target_boxes.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # Get output of model
        out_pts, out_bbox, out_offsets = model(inputs)
        # out_pts = model(inputs)

        center_loss = criterion(out_pts, target_pts)
        if args.sr_opt[0] != "none":
            sr_term_1 = get_SR_term(out_pts, args.sr_opt[0], target_pts, criterion)

        targets_aabb_min_pts = target_boxes[:, :, 0, :]
        targets_aabb_min_pts = targets_aabb_min_pts.reshape(-1, 64)
        targets_aabb_max_pts = target_boxes[:, :, 1, :]
        targets_aabb_max_pts = targets_aabb_max_pts.reshape(-1, 64)

        target_hw = targets_aabb_max_pts - targets_aabb_min_pts
        target_hw = target_hw.to(device)
        bbox_loss = criterion(out_bbox, target_hw)

        # Third Loss
        if args.target == "offset":
            target_offsets = target_pts - out_pts

            offset_loss = criterion(out_offsets, target_offsets)

            # Add spatial regularization
            if args.sr_opt[1] != "none":
                sr_term_2 = get_SR_term(out_pts + out_offsets, args.sr_opt[1], target_pts, criterion)

        elif args.target == "center":
            target_origins = out_pts.clone().detach()
            patch_size = list(model.module.get_patch_size())

            target_origins[:, 0::2] -= patch_size[0] / 2
            target_origins[:, 1::2] -= patch_size[1] / 2

            target_centers = target_pts.clone().detach()
            target_centers = target_centers - target_origins

            offset_loss = criterion(out_offsets, target_centers)

            # Add spatial regularization
            if args.sr_opt[1] != "none":
                sr_term_2 = get_SR_term(target_origins + out_offsets, args.sr_opt[1], target_pts, criterion)

        elif args.target == "none":
            pass

        else:
            raise ValueError('Not allowed loss_target option')

        total_loss = args.weight[0] * center_loss + args.weight[1] * bbox_loss

        if args.target != "none":
            total_loss = total_loss + args.weight[2] * offset_loss

        if args.sr_opt[0] != "none":
            total_loss = total_loss + args.sr_weight[0] * sr_term_1
        if args.sr_opt[1] != "none":
            total_loss = total_loss + args.sr_weight[1] * sr_term_2

        total_loss.backward()
        optimizer.step()

        # Calculate for logging
        batch_size = inputs.size(0)
        train_center_loss += center_loss.item() * batch_size
        train_bbox_loss += bbox_loss.item() * batch_size
        if args.target != "none":
            train_offset_loss += offset_loss.item() * batch_size
        if args.sr_opt[0] != "none":
            train_sr_term_1 += sr_term_1.item() * batch_size
        if args.sr_opt[1] != "none":
            train_sr_term_2 += sr_term_2.item() * batch_size
        train_loss += total_loss.item() * batch_size

        num_inputs += batch_size

        # Progress bar
        data_cnt = len(data_loader.dataset)
        done_cnt = min((batch_idx + 1) * data_loader.batch_size, data_cnt)
        rate = done_cnt / data_cnt
        bar = ('=' * int(rate * 32) + '>').ljust(32, '.')
        idx = str(done_cnt).rjust(len(str(data_cnt)), ' ')
        print('\rTrain\t : {}/{}: [{}]'.format(
            idx,
            data_cnt,
            bar
        ), end='')

    train_center_loss = train_center_loss / num_inputs
    train_bbox_loss = train_bbox_loss / num_inputs
    train_offset_loss = train_offset_loss / num_inputs
    train_sr_term_1 = train_sr_term_1 / num_inputs
    train_sr_term_2 = train_sr_term_2 / num_inputs

    train_loss = train_loss / num_inputs

    print()
    print("Center Loss(MSE): %f"    % train_center_loss)
    print("Bbox Loss(MSE): %f"      % train_bbox_loss)
    print("Offset Loss(MSE): %f"    % train_offset_loss)
    print("SR Term: %f"             % train_sr_term_1)
    print("SR Term: %f"             % train_sr_term_2)
    print("Total Train Loss: %f"    % train_loss)
    print()
    writer.add_scalars("Loss/train", {
        "center_loss" : train_center_loss,
        "bbox_loss" : train_bbox_loss,
        "offset_loss" : train_offset_loss,
        "sr_term_1" : train_sr_term_1,
        "sr_term_2": train_sr_term_2,
        "total_loss": train_loss,
    }, epoch)


# Validate
min_val_loss = float("inf")
def evaluate(model, epoch, data_loader, criterion, scheduler, writer, device, args):
    assert('target' in args)

    model.eval()

    val_center_loss = 0.0
    val_bbox_loss = 0.0
    val_offset_loss = 0.0
    val_sr_term_1 = 0.0
    val_sr_term_2 = 0.0
    val_loss = 0.0
    num_inputs = 0

    global min_val_loss

    with torch.no_grad():
        for batch_idx, (inputs, target_pts, target_boxes) in enumerate(data_loader):
            inputs, target_pts, target_boxes = \
                inputs.to(device), target_pts.to(device), target_boxes.to(device)

            # Get output of model
            out_pts, out_bbox, out_offsets = model(inputs)

            center_loss = criterion(out_pts, target_pts)

            if args.sr_opt[0] != "none":
                sr_term_1 = get_SR_term(out_pts, args.sr_opt[0], target_pts, criterion)

            targets_aabb_min_pts = target_boxes[:, :, 0, :]
            targets_aabb_min_pts = targets_aabb_min_pts.reshape(-1, 64)
            targets_aabb_max_pts = target_boxes[:, :, 1, :]
            targets_aabb_max_pts = targets_aabb_max_pts.reshape(-1, 64)

            target_hw = targets_aabb_max_pts - targets_aabb_min_pts
            target_hw = target_hw.to(device)
            bbox_loss = criterion(out_bbox, target_hw)

            # Third Loss
            if args.target == "offset":
                target_offsets = target_pts - out_pts

                offset_loss = criterion(out_offsets, target_offsets)

                # Add spatial regularization
                if args.sr_opt[1] != "none":
                    sr_term_2 = get_SR_term(out_pts + out_offsets, args.sr_opt[1], target_pts, criterion)

            elif args.target == "center":
                target_origins = out_pts.clone().detach()
                patch_size = list(model.module.get_patch_size())

                target_origins[:, 0::2] -= patch_size[0] / 2
                target_origins[:, 1::2] -= patch_size[1] / 2

                target_centers = target_pts.clone().detach()
                target_centers = target_centers - target_origins

                offset_loss = criterion(out_offsets, target_centers)

                # Add spatial regularization
                if args.sr_opt[1] != "none":
                    sr_term_2 = get_SR_term(out_pts + out_offsets, args.sr_opt[1], target_pts, criterion)

            elif args.target == "none":
                pass

            else:
                raise ValueError('Not allowed loss_target option')

            total_loss = args.weight[0] * center_loss + args.weight[1] * bbox_loss

            if args.target != "none":
                total_loss = total_loss + args.weight[2] * offset_loss

            if args.sr_opt[0] != "none":
                total_loss = total_loss + args.sr_weight[0] * sr_term_1
            if args.sr_opt[1] != "none":
                total_loss = total_loss + args.sr_weight[1] * sr_term_2

            batch_size = inputs.size(0)
            val_center_loss += center_loss.item() * batch_size
            val_bbox_loss += bbox_loss.item() * batch_size
            if args.target != "none":
                val_offset_loss += offset_loss.item() * batch_size
            if args.sr_opt[0] != "none":
                val_sr_term_1 += sr_term_1.item() * batch_size
            if args.sr_opt[1] != "none":
                val_sr_term_2 += sr_term_2.item() * batch_size
            val_loss += total_loss.item() * batch_size
            num_inputs += batch_size

            # Progress bar
            data_cnt = len(data_loader.dataset)
            done_cnt = min((batch_idx + 1) * data_loader.batch_size, data_cnt)
            rate = done_cnt / data_cnt
            bar = ('=' * int(rate * 32) + '>').ljust(32, '.')
            idx = str(done_cnt).rjust(len(str(data_cnt)), ' ')
            print('\rVal\t : {}/{}: [{}]'.format(
                idx,
                data_cnt,
                bar
            ), end='')

        val_center_loss = val_center_loss / num_inputs
        val_bbox_loss = val_bbox_loss / num_inputs
        val_offset_loss = val_offset_loss / num_inputs
        val_sr_term_1 = val_sr_term_1 / num_inputs
        val_sr_term_2 = val_sr_term_2 / num_inputs

        val_loss = val_loss / num_inputs

        print()
        print("Center Loss(MSE): %f" % val_center_loss)
        print("Bbox Loss(MSE): %f" % val_bbox_loss)
        print("Offset Loss(MSE): %f" % val_offset_loss)
        print("SR Term: %f" % val_sr_term_1)
        print("SR Term: %f" % val_sr_term_2)
        print("Total Val Loss: %f" % val_loss)
        writer.add_scalars("Loss/val", {
            "center_loss": val_center_loss,
            "bbox_loss": val_bbox_loss,
            "offset_loss": val_offset_loss,
            "sr_term_1": val_sr_term_1,
            "sr_term_2": val_sr_term_2,
            "total_loss": val_loss,
        }, epoch)

        scheduler.step(val_loss)

    # Save checkpoint
    # state = {
    #     'name': args.prj_name,
    #     'backbone': args.backbone,
    #     'epoch': epoch,
    #     'image_size': args.image_size,
    #     'patch_size': args.patch_size,
    #     'loss_target': args.target,
    #     'net': model.state_dict()
    # }
    # torch.save(state, os.path.join(args.model_save_path, "ckpt-" + str(epoch) + ".pth"))
    if val_loss < min_val_loss:
        state = {
            'name': args.prj_name,
            'backbone': args.backbone,
            'epoch': epoch,
            'image_size': args.image_size,
            'patch_size': args.patch_size,
            'loss_target': args.target,
            'net': model.state_dict()
        }
        torch.save(state, os.path.join(args.model_save_path, "best.pth"))

        min_val_loss = val_loss


def main():
    # Parsing argument
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)

    parser.add_argument('--prj_name', type=str, default='', help='project name')
    parser.add_argument('--batch_size', type=int, default=12, help='batch size (default: 12)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning_rate (default: 1e-3)')
    parser.add_argument('--max_epoch', type=int, default=300, help='maximum epoch (default: 300) ')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--model_save_path', required=True,
                        help='Directory path to save model checkpoints')
    parser.add_argument('--log_path', required=True,
                        help='Directory path to save model logs')
    parser.add_argument('--data_path', required=True,
                        help='Directory path for train and validation data')
    parser.add_argument('--pretrained_model_path', required=True,
                        help='Directory path for Pre-trained model')

    # Training options
    parser.add_argument('--use_pretrained', action='store_true', default=False,
                        help='Flag to use pretrained backbone')
    parser.add_argument('--backbone', type=str, choices=['resnet', 'dla', 'hourglass'], default='resnet',
                        help='')
    parser.add_argument('--image_size', type=int, nargs=2, default=(512, 768),
                        help='model input image size (default: (512, 768))')
    parser.add_argument('--patch_size', type=int, nargs=2, default=(138, 92),
                        help='patch_size (default: (138, 92))')
    parser.add_argument('--weight', type=float, nargs=3, default=(1, 1.5, 3),
                        metavar=('WEIGHT1', 'WEIGHT2', 'WEIGHT3'),
                        help='weight of losses (default: (1, 1.5, 3))\n'
                             'center_loss\n'
                             'bbox_loss\n'
                             'offset_loss')
    parser.add_argument('--target',  choices=['none', 'offset', 'center'], default='offset',
                        help="Train center or offset after cropping patches.")
    parser.add_argument('--sr_opt', nargs=2, choices=['none', 'mse', 'sd_mse', 'l2'], default=('none', 'none'),
                        help='Opts for Spatial Regulariations\n'
                             'none\t:\n'
                             'mse\t:\n'
                             'sd_mse\t:\n'
                             'l2\t:')
    parser.add_argument('--sr_weight', nargs=2, type=float, default=(0.1, 0.1),
                        help='weight for spatial regularization loss')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')

    # Tensorboard
    writer = SummaryWriter(log_dir=args.log_path)

    # Model
    model = Pano(image_size=args.image_size, patch_size=args.patch_size, backbone=args.backbone, target=args.target)
    model = model.to(device)

    model = nn.DataParallel(model).to(device)

    # # 모델의 state_dict 출력
    # print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Load backbone model
    if args.use_pretrained is True:
        if args.backbone == "resnet":
            checkpoint = torch.load(os.path.join(args.pretrained_model_path, "backbone-resnet.pth"))
        elif args.backbone == "dla":
            checkpoint = torch.load(os.path.join(args.pretrained_model_path, "backbone-dla.pth"))
        elif args.backbone == "hourglass":
            checkpoint = torch.load(os.path.join(args.pretrained_model_path, "backbone-hg.pth"))
        else:
            raise ValueError

        # Alert
        print()
        print("Loaded pre-trained model")

        model.module.backbone.load_state_dict(checkpoint['net'])

    criterion = nn.MSELoss().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # lr scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, verbose=True, eps=1e-10)

    # Data
    print("Build DataLoader...", end="")
    train_dataset = TeethDataset(path=os.path.join(args.data_path, 'train'))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2,
                                               pin_memory=True, drop_last=True)
    val_dataset = TeethDataset(path=os.path.join(args.data_path, 'val'))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
                                             pin_memory=True)
    print("Done")

    # For debugging arguments
    print(args)

    train_begin = time.time()

    for epoch in range(args.max_epoch):
        train(model, epoch, train_loader, criterion, optimizer, writer, device, args)
        evaluate(model, epoch, val_loader, criterion, scheduler, writer, device, args)

    writer.close()

if __name__ == "__main__":
    # torch.set_printoptions(profile="full")
    main()
