import argparse

from train import train
from run import run

def get_args():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## For train
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batch_size')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-w', '--weights', nargs='*',
                        help='Class weights to use in loss calculation')
    parser.add_argument('--n_channels', type=int, default=3,
                        help='Number of channels in input images')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')


    ## for both train and inference
    parser.add_argument('-d', '--mode', dest='mode', type=str, default='train',
                        help='Mode to run the U-Net, train or infer')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('--n_classes', type=int, default=1,
                        help='Number of classes in the segmentation')


    ## for inference
    parser.add_argument('-m', '--model', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('-i', '--input', metavar='INPUT', default='/data/test/',
                        help='Folder to read input images')

    parser.add_argument('-o', '--output', metavar='OUTPUT', default='./output/',
                        help='Folder to save ouput images')
    parser.add_argument('-n', '--no-save', action='store_true', default=False,
                        help='Do not save the output masks')
    parser.add_argument('-t', '--mask-threshold', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')


    return parser.parse_args()



if __name__ == '__main__':
    args = get_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'infer':
        run(args)
    else:
        print('mode must train or test')
