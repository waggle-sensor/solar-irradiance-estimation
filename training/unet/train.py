import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

from dice_loss import dice_coeff

import json

dir_img = '/data/train/images/'
dir_mask = '/data/train/labels/'
dir_checkpoint = 'checkpoints/'


def train_net(args,
              net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5,
              n_classes=2,
              class_weights = [1,1],):

    dataset = BasicDataset(dir_img, dir_mask, img_scale, n_classes)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    args.training_size = n_train
    args.validation_size = n_val

    argparse_dict = vars(args)
    with open('./runs/config.list', 'w') as f:
        f.write(json.dumps(argparse_dict, indent=4, sort_keys=True))

    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to(device=device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    if os.path.exists('./runs/log.csv'):
        os.remove('./runs/log.csv')
    f = open('./runs/log.csv', 'a')

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)

                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                tot = 0.
                for true_mask, pred in zip(true_masks, masks_pred):
                    pred = (pred > 0.5).float()
                    tot += dice_coeff(pred, true_mask.squeeze(dim=1)).item()
                acc = tot / len(imgs)

                writer.add_scalar('Acc/train', acc, global_step)


                log = [epoch] + [loss.item()] + [acc] + [' ']
                log = map(str, log)
                f.write(','.join(log) + '\n')


                pbar.update(imgs.shape[0])
                global_step += 1

                if global_step % (len(dataset) // (10 * batch_size)) == 0:
                    val_score = eval_net(net, val_loader, device, n_val)
                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/val', val_score, global_step)

                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Acc/val', val_score, global_step)

                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

                    log = [epoch] + [' '] + [' '] + [val_score]
                    log = map(str, log)
                    f.write(','.join(log) + '\n')

                    net.train()

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()

'''
    ('-e', '--epochs', metavar='E', type=int, default=5, help='Number of epochs', dest='epochs')
    ('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1, help='Batch size', dest='batch_size')
    ('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.00001, help='Learning rate', dest='lr')
    ('-w', '--weights', nargs='*', help='Class weights to use in loss calculation')
    ('--n_classes', type=int, default=1, help='Number of classes in the segmentation')
    ('--n_channels', type=int, default=3, help='Number of channels in input images')
    ('-f', '--load', dest='load', type=str, default=False, help='Load model from a .pth file')
    ('-s', '--scale', dest='scale', type=float, default=0.5, help='Downscaling factor of the images')
    ('-v', '--validation', dest='val', type=float, default=10.0, help='Percent of the data that is used as validation (0-100)')
'''

def train(args):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    #args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.cuda = torch.cuda.is_available()
    logging.info(f'Using device {device}')

    n_classes = args.n_classes
    n_channels = args.n_channels
    class_weights = np.array(args.weights).astype(np.float)
    #assert len(class_weights) == n_classes, \
    #    'Lenght of the weights-vector should be equal to the number of classes'

    net = UNet(n_channels=3, n_classes=n_classes)



    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Dilated conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(args,
                  net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  n_classes=n_classes,
                  class_weights = class_weights,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
