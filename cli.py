import argparse
import os

import torch
import math
def str_or_float(value):
    try:
        return float(value)
    except ValueError:
        return value

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--dev', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='cpu or cuda')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')
    parser.add_argument('--ebd_dim', type=int, default=32,
                        help='dimension of embedded features.')
    parser.add_argument('--cp_load_path', type=str, default='no',
                        help='path to the checkpoint. no means to train from scratch.')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'mlp'], help='structure of the embedding network')
    parser.add_argument('--dataset_path', type=str, default='RiverPIXELS/data_riverpixels.npy',
                        help='path to the dataset .npy or .npz file.')
    parser.add_argument('--cloud_augmentation', action='store_true',
                        help='use the cloud augmentation.')

    # parser.add_argument('--visualize_dataset_path', type=str, default='RiverPIXELS/data_riverpixels.npy',
    #                     help='path to the dataset .npy or .npz file for the low-dim visualize task.')
    # parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    # parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR', 'Val'], help='choose method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.5,
                        help='temperature for loss function')

    # for dataset
    parser.add_argument('--margin', type=int, default=4,
                        help='the margin size of NLM patches')
    parser.add_argument('--stepsize', type=int, default=1,
                        help='the stepsize of dataset')
    parser.add_argument('--down_sample_rate', type=str, default='0.03,0.2,1',
                        help='data downsample rates')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')

    opt = parser.parse_args()

    opt.model_path = f"./save/{opt.method}"

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    if opt.down_sample_rate == "None":
        opt.ratios = None
    else:
        ratios = opt.down_sample_rate.split(',')
        opt.ratios = list([])
        for rate in ratios:
            opt.ratios.append(float(rate))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    # if opt.batch_size > 256:
    #     opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt

def parse_option_TestAndEvaluate():
    parser = argparse.ArgumentParser('argument for active learning selection')

    parser.add_argument('--train_dataset_path', type=str, default='RiverPIXELS/Fix_features/AAL_features.npy',
                        help='path to the training dataset (.npy or .npz) file.')
    parser.add_argument('--test_dataset_path', type=str, default="No",
                        help='path to the test dataset (.npy, .npz or .tif) file. Also can be "No" to use GEE data')
    parser.add_argument('--sample_batch_size', type=int, default=65536,
                        help='the batch size of the sample process')
    parser.add_argument('--random_sample', action='store_true',
                        help='random sample data from the test set')
    parser.add_argument('--show_figures', action='store_true',
                        help='show the prediction label figures.')
    parser.add_argument('--ssl_method', type=str, default='Stable_Laplace',
                        choices=['Laplace', 'rw_Laplace', 'Poisson', 'Stable_Laplace'],
                        help='use the percentage of training set as the class prior.')
    parser.add_argument('--figure_save_path', type=str, default='auto',
                        help='The path that result figures are saved to')

    ## for custom dataset only
    parser.add_argument('--dev', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='cpu or cuda')
    parser.add_argument('--margin', type=int, default=4,
                        help='the margin size of NLM patches')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='num of workers to use')
    parser.add_argument('--ebd_dim', type=int, default=32,
                        help='dimension of embedded features.')
    parser.add_argument('--cp_load_path', type=str, default='save/SupCon/SupCon.pth',
                        help='path to the checkpoint. no means to train from scratch.')
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'mlp'], help='structure of the embedding network')
    parser.add_argument('--test_dataset_type', type=str, default='Landsat 5',
                        help='Type of test dataset')

    # for GEE data only
    parser.add_argument('--lon_lat', type=str, default='auto',
                        help='the lon and lat values, should be input as "lon, lat". Can be set as "auto" to use ChatGPT.')
    parser.add_argument('--location', type=str, default='Ucayali River',
                        help='when --lon_lat is "auto", use --location to get the lon and lat values.')
    parser.add_argument('--ee_dataset', type=str, default='LANDSAT/LT05/C01/T1_TOA',
                        help='the dataset name in GEE.')
    parser.add_argument('--startdate', type=str, default='2010-01-01',
                        help='the start date of filter YYYY-MM-DD')
    parser.add_argument('--enddate', type=str, default='2010-12-31',
                        help='the end date of filter YYYY-MM-DD')
    parser.add_argument('--lon_range', type=float, default=0.25,
                        help='the range of longitude of rectangle')
    parser.add_argument('--lat_range', type=float, default=0.25,
                        help='the range of latitude of rectangle')
    parser.add_argument('--display_image', action='store_true',
                        help='display the gee image')
    parser.add_argument('--gee_save_name', type=str, default='Landsat5_Image',
                        help='save name of the gee image')
    parser.add_argument('--scale', type=int, default=30,
                        help='the resolution of the image')
    parser.add_argument('--checkfile_retries', type=int, default=3,
                        help='the time to check if the GEE image is downloaded.')
    parser.add_argument('--checkfile_waiting_time', type=int, default=90,
                        help='the seconds to wait between each retry of the file checking.')


    opt = parser.parse_args()

    return opt