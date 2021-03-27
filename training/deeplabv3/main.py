import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils import data

import network
import utils
from metrics import StreamSegMetrics
from datasets.waggle_cloud import WaggleSegmentation
from deeplabv3 import DeepLab

import glob
import imageio

class Trainer():
    def __init__(self, data_loader, opts):
        #super(Trainer, self).__init__(data_loader, opts)
        #self.opts = opts
        self.train_loader = data_loader[0]
        self.val_loader = data_loader[1]

        # Set up model
        model_map = {
            'deeplabv3_resnet50': network.deeplabv3_resnet50,
            'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
            'deeplabv3_resnet101': network.deeplabv3_resnet101,
            'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
            'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
            'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
        }

        self.model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
        if opts.separable_conv and 'plus' in opts.model:
            network.convert_to_separable_conv(self.model.classifier)

        def set_bn_momentum(model, momentum=0.1):
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.momentum = momentum

        set_bn_momentum(self.model.backbone, momentum=0.01)             ##### What is Momentum? 0.01 or 0.99? #####


        # Set up metrics
        self.metrics = StreamSegMetrics(opts.num_classes)

        # Set up optimizer
        self.optimizer = torch.optim.SGD(params=[
            {'params': self.model.backbone.parameters(), 'lr': 0.1*opts.lr},
            {'params': self.model.classifier.parameters(), 'lr': opts.lr},
        ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

        if opts.lr_policy=='poly':
            self.scheduler = utils.PolyLR(self.optimizer, opts.total_itrs, power=0.9)
        elif opts.lr_policy=='step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opts.step_size, gamma=0.1)

        # Set up criterion
        if opts.loss_type == 'focal_loss':
            self.criterion = utils.FocalLoss(ignore_index=255, size_average=True)
        elif opts.loss_type == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')


        self.best_mean_iu = 0
        self.iteration = 0


    def _label_accuracy_score(self, label_trues, label_preds, n_class):
        """Returns accuracy score evaluation result.
          - overall accuracy
          - mean accuracy
          - mean IU
          - fwavacc
        """
        def _fast_hist(label_true, label_pred, n_class):
            mask = (label_true >= 0) & (label_true < n_class)
            hist = np.bincount(n_class * label_true[mask].astype(int) +
                           label_pred[mask],
                           minlength=n_class**2).reshape(n_class, n_class)
            return hist



        hist = np.zeros((n_class, n_class))
        for lt, lp in zip(label_trues, label_preds):
            hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, mean_iu, fwavacc


    def validate(self, opts):
        # import matplotlib.pyplot as plt
        training = self.model.training
        self.model.eval()

        n_class = opts.num_classes

        val_loss = 0
        visualizations = []
        label_trues, label_preds = [], []
        with torch.no_grad():
            for i, (data, target) in tqdm(enumerate(self.val_loader)):
                data, target = data.to(opts.device, dtype=torch.float), target.to(opts.device)
                score = self.model(data)

                loss = self.criterion(score, target)
                if np.isnan(float(loss.item())):
                    raise ValueError('loss is nan while validating')
                val_loss += float(loss.item()) / len(data)

                imgs = data.data.cpu()
                lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
                lbl_true = target.data.cpu()
                for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
                    img, lt = self.val_loader.dataset.untransform(img, lt)
                    label_trues.append(lt)
                    label_preds.append(lp)
                    if len(visualizations) < 9:
                        pass
                        viz = utils.fcn_utils.visualize_segmentation(lbl_pred=lp,
                                                           lbl_true=lt,
                                                           img=img,
                                                           n_class=opts.num_classes)
                        visualizations.append(viz)
                        pass
        acc, acc_cls, mean_iu, fwavacc = self._label_accuracy_score(label_trues, label_preds, n_class)

        out = os.path.join(opts.output, 'visualization_viz')
        if not os.path.exists(out):
            os.makedirs(out)
        out_file = os.path.join(out, 'iter%012d.jpg' % self.iteration)
        #raise Exception(len(visualizations))
        img_ = utils.fcn_utils.get_tile_image(visualizations)
        imageio.imwrite(out_file, img_)
        # plt.imshow(imageio.imread(out_file))
        # plt.show()

        val_loss /= len(self.val_loader)



        is_best = mean_iu > self.best_mean_iu
        if is_best:  # save best model
            self.best_mean_iu = mean_iu

        def save_ckpt(path):
            """ save current model
            """
            torch.save({
                "cur_itrs": self.iteration,
                "model_state": self.model.module.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "best_score": self.best_mean_iu,
            }, path)
            print("Model saved as %s" % path)


        if is_best:
            save_ckpt('checkpoints/best_%s_%s_os%d.pth' % (opts.model, opts.dataset, opts.output_stride))
        else:
            save_ckpt('checkpoints/latest_%s_%s_os%d.pth' % (opts.model, opts.dataset, opts.output_stride))

        if training:
            self.model.train()

        """Do validation and return specified samples"""
        self.metrics.reset()




    def train(self, opts):

        print("Dataset: %s, Train set: %d, Val set: %d" % (opts.dataset, len(self.train_loader), len(self.val_loader)))


        if not os.path.exists('./checkpoints'):
            os.mkdir('./checkpoints')

        # Restore
        best_score = 0.0
        cur_itrs = 0
        cur_epochs = 0
        if opts.ckpt is not None and os.path.isfile(opts.ckpt):
            print('[!] Retrain from a checkpoint')
            # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
            checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint["model_state"])
            self.model = nn.DataParallel(self.model)
            self.model.to(opts.device)
            if opts.continue_training:
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
                self.scheduler.load_state_dict(checkpoint["scheduler_state"])
                cur_itrs = checkpoint["cur_itrs"]
                best_score = checkpoint['best_score']
                print("Training state restored from %s" % opts.ckpt)
            print("Model restored from %s" % opts.ckpt)
            del checkpoint  # free memory
        else:
            print('[!] Retrain from base network')
            self.model = nn.DataParallel(self.model)
            self.model.to(opts.device)

        #==========   Train Loop   ==========#
        interval_loss = 0
        while True: #cur_itrs < opts.total_itrs:
            # =====  Train  =====
            self.model.train()
            cur_epochs += 1
            for (images, labels) in self.train_loader:
                cur_itrs += 1

                images = images.to(opts.device, dtype=torch.float32)
                labels = labels.to(opts.device, dtype=torch.long)

                self.optimizer.zero_grad()

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                np_loss = loss.detach().cpu().numpy()
                interval_loss += np_loss


                lbl_pred = outputs.data.max(1)[1].cpu().numpy()[:, :, :]
                lbl_true = labels.data.cpu().numpy()
                acc, acc_cls, mean_iu, fwavacc = self._label_accuracy_score(lbl_true, lbl_pred, n_class=opts.num_classes)




                self.iteration = cur_itrs
                '''
                if (cur_itrs) % 10 == 0:
                    interval_loss = interval_loss/10
                    print("Epoch %d, Itrs %d/%d, Loss=%f" %
                          (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                    interval_loss = 0.0
                '''

                if cur_itrs % opts.interval_val == 0:
                    print('Epoch %d, Itrs %d/%d' % (cur_epochs, cur_itrs, opts.total_itrs))
                    self.validate(opts)

                if cur_itrs >=  opts.total_itrs:
                    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train', help='purpose of the run', choices=['train', 'val'])

    # Datset Options
    parser.add_argument('--input_path', type=str, help='input dataset path')
    parser.add_argument('--dataset', type=str, help='Name of dataset', choices=['cityscapes', 'waggle_cloud', 'voc'])
    parser.add_argument('--num_classes', type=int, default=21, help='number of classes (default: None)')
    parser.add_argument('--resize', type=int, help='resize image')

    # Deeplab Options
    parser.add_argument('--separable_conv', action='store_true', default=False, help='apply separable conv to decoder and aspp')
    parser.add_argument('--output_stride', type=int, default=16, choices=[8, 16])
    parser.add_argument('--model', type=str, default='deeplabv3_resnet50', help='model name', choices=['deeplabv3_resnet50', 'deeplabv3plus_resnet50', 'deeplabv3_resnet101', 'deeplabv3plus_resnet101', 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'])
    # Train Options
    parser.add_argument('--output', type=str, default='./output', help='folder for output images path')
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--n_workers', type=int, default=4, help='number of workers to read dataset')

    parser.add_argument('--total_itrs', type=int, default=30e3, help='epoch number (default: 30k)')
    parser.add_argument('--lr', type=float, default=0.000001, help='learning rate (default: 0.000001)')
    parser.add_argument('--lr_policy', type=str, default='poly', help='learning rate scheduler policy', choices=['poly', 'step'])
    parser.add_argument('--step_size', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=16, help='batch size (default: 16)')
    parser.add_argument('--interval_val', type=int, default=1000, help='iteration interval for validation')

    parser.add_argument('--ckpt', default=None, type=str, help='restore from checkpoint')
    parser.add_argument('--continue_training', action='store_true', default=False)

    parser.add_argument('--loss_type', type=str, default='cross_entropy', help='loss type (default: cross_entropy)', choices=['cross_entropy', 'focal_loss'])
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-4, help='weight decay (default: 1e-4)')

    args = parser.parse_args()

    if args.mode == 'val':
        if args.input_path == None or args.ckpt == None or args.model == None or args.num_classes == None:
            parser.print_help()
            print('[ERROR] --input_path, --ckpt, --model, --num_classes are required for validation')
            exit(0)


        deeplab = DeepLab(args)

        if 'jpg' in args.input_path or 'png' in args.input_path or 'jpeg' in args.input_path:
            print('a')
            deeplab.run(args.input_path)
        elif 'txt' in args.input_path:
            import shutil
            print('b')
            count = 0
            dir = os.path.dirname(args.input_path).split('I')[0] + 'JPEGImages/'
            print(dir)
            with open(args.input_path, 'r') as f:
                for line in f:
                    count += 1
                    path = os.path.join(dir, line.strip() + '.jpg')
                    shutil.copy2(path, './images')
                    deeplab.run(path)
                    if count == 100:
                        exit(0)
        else:
            dir = os.path.join(args.input_path, '*')
            inputs = glob.glob(dir)
            print(len(inputs))
            for i in inputs:
                deeplab.run(i)

        exit(0)



    if args.dataset == None:
        parser.print_help()
        exit(0)
    elif args.dataset == 'waggle_cloud' and args.input_path == None:
        parser.print_help()
        print('[ERROR] --input_path is required')
        exit(1)
    elif args.dataset == 'voc' and args.input_file == None:
        parser.print_help()
        print('[ERROR] --input_file is required')
        exit(1)
    elif args.batch_size < 3:
        print('[ERROR] --batch_size must be greater than 1 (greater than 2 -- when I test in Alien, it spits the same error; torch shape does not match becuase of small batch size)')
        print('[INFO] in training, there must be more than 1 sample unless you set drop_last=True in your Dataloader')
        exit(1)



    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cuda = torch.cuda.is_available()
    kwargs = {'num_workers': args.n_workers} if cuda else {}
    train_loader = torch.utils.data.DataLoader(
        WaggleSegmentation(
            args,
            image_set='train',
            transform=True),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )

    val_loader = torch.utils.data.DataLoader(
        WaggleSegmentation(
            args,
            image_set='val',
            transform=True),
        batch_size=args.batch_size,
        shuffle=False,
        **kwargs
    )
    data_loader = [train_loader, val_loader]


    #train_dst = WaggleSegmentation(opts, image_set='train', transform=True)
    #val_dst = WaggleSegmentation(opts, image_set='val', transform=False)


    trainer = Trainer(data_loader, args)
    trainer.train(args)
    #main(args, train_loader, val_loader)

