#%% Tensorboard setep
# (1) tensorboard --logdir="...<ThisRepoFolder>\runs"
# (2) http://localhost:6006

#%% Imports

import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import yaml

#from training.train import train_model
#from inference.evaluate import evaluate_model
#from data.prepare_clips import prepare_data

#%% Initialize


#%% Methods

def parse_args():
    ''' Construct parser
    '''
    parser = argparse.ArgumentParser(description = '3D-AE via PyTorch')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--mode', type=str, choices=["prep","train","eval"], required=True,
                        help='set mode to evaluate or train model')
    parser.add_argument('-data', metavar='DIR', default = f'{os.getcwd()}\data',
                        help='path to dataset')
    parser.add_argument('-dataset-name', default='example_data_folder.tar',
                        help='dataset name', choices = os.listdir(f'{os.getcwd()}\data'))
    parser.add_argument('--workers', default=4, type=int,
                        help='number of workers for dataloader (default: 4)')
    parser.add_argument('--gpu-index', default=0, type=int,
                        help='Use Gpu-index(0) or not if (-1)/None (default: 0).')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                        help='size of image batches per epoch')
    parser.add_argument('--image-size', default=224, type=int,
                        help='output image size once processed by transforms for dataloader (default: 224)')

    return parser.parse_args()

def main():
    ''' Main operating script
    '''

    args = parse_args()

    # check if gpu training is available
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        num_workers = min(args.workers, os.cpu_count())
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        num_workers = 0 # keep safe for CPU-only, avoid multiprocessing issues
        args.gpu_index = -1
        
    print("Device:", args.device)
    print("Number of workers assigned:", num_workers)
    # print("Total number of epochs:", args.epochs)
    # print("Number of warmup epochs:", args.warmup_epochs)

    # Check mode and proceed accordingly
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)  # load config file into local var

    #if args.mode == "prep":
    #    prepare_data(config)
    #elif args.mode == "train":
    #    train_model(config)
    #elif args.mode == "eval":
    #    evaluate_model(config)

    pass

#%% Script
if __name__ == "__main__":  # only operate script if called directly (i.e., not as module, __name__ = "my_module")
    
    main()

