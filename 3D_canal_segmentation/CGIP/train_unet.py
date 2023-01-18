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

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Training
def train(model, epoch, data_loader, criterion, optimizer, writer, device, args):
    model.train()

    print('\n==> epoch %d' % epoch)
    train_loss = 0.0
    num_inputs = 0

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        # zero the parameter gradients
        optimizer.zero_grad()

        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs).squeeze() # outputs: (b, d, h, w)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        num_inputs += args.batch_size

    train_loss /= num_inputs

    print("------Train------")
    print("Unet Train Loss: %f"    % train_loss)
    print()
    writer.add_scalars("Loss/train", {
        "train_loss": train_loss,
    }, epoch)


# Validate
def evaluate(model, epoch, data_loader, criterion, writer, device, args):
    model.eval()

    val_loss = 0.0
    num_inputs = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            # zero the parameter gradients
            optimizer.zero_grad()

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()  # outputs: (b, d, h, w)

            loss = criterion(outputs, targets)

            val_loss += loss.item()
            num_inputs += args.batch_size

        val_loss /= num_inputs

        print("------Val------")
        print("Unet val Loss: %f"    % val_loss)
        print()
        writer.add_scalars("Loss/val", {
            "val_loss": val_loss,
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
    parser.add_argument('--batch_size', type=int, default=8, help='batch size (default: 2)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning_rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight_decay (default: 1e-5)')
    parser.add_argument('--max_epoch', type=int, default=300, help='maximum epoch (default: 300) ')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')

    # Training options
    parser.add_argument('--crop_info', type=int, nargs=6, default=(96, 416, 0, 320, -320, 0),
                        help='model input image crop info size(x, x, y, y, z, z) (default: (96, 416, 0, 320, -320, 0))')
    parser.add_argument('--resume', type=int, default=0, help='resume from checkpoint')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if args.cuda else 'cpu')

    args.prj_name = 'unet_mini_bce'
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

    args.resume = 0
    args.include_wisdom = True
    num_tooth_type = 8 if args.include_wisdom else 7
    args.num_class = num_tooth_type * 4
    tooth_list = []
    for i in range(args.num_class):
        tooth_num = (i // num_tooth_type) * 10 + i % num_tooth_type + 11
        tooth_list.append(tooth_num)

    # Tensorboard
    writer = SummaryWriter(log_dir=args.log_path)

    # Model
    net = UNet_3D(n_class=1)

    # load checkpoint
    if args.resume > 0:
        checkpoint = torch.load(os.path.join(args.model_save_path, 'ckpt-' + str(args.resume) + '.pth'))
        net.load_state_dict(checkpoint['net'], strict=True)

    net = net.to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print("Model Done")

    # Data
    train_dataset = MaskDataset(
        mode='train',
        img_path=os.path.join(args.data_path, 'train', 'image_big'),
        gt_path=os.path.join(args.data_path, 'train', 'gt_dist_map'),
        meta_path=args.meta_path,
        crop=args.crop_info,
        tooth_list=tooth_list
    )
    val_dataset = MaskDataset(
        mode='val',
        img_path=os.path.join(args.data_path, 'val', 'image_big'),
        gt_path=os.path.join(args.data_path, 'val', 'gt_dist_map'),
        meta_path=args.meta_path,
        crop=args.crop_info,
        tooth_list=tooth_list
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print("Dataset Done", args.num_class)

    # run
    for epoch in range(args.resume, args.max_epoch + 1):
        train(net, epoch, train_loader, criterion, optimizer, writer, device, args)
        evaluate(net, epoch, val_loader, criterion, writer, device, args)

    writer.close()
