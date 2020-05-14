import argparse
import os
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.quantization
from torch.utils.data import DataLoader

from dataset import LivenessDataset
from helpers import *
from logger import create_logger
from model import CustomCNN


def load_model(model_file, args):
    model = CustomCNN(args)
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    model.to('cpu')
    return model


def get_args(parser):
    parser.add_argument("--pos_path", type=str, default="/Users/suvrat/Downloads/suv/real")
    parser.add_argument("--neg_path", type=str, default="/Users/suvrat/Downloads/suv/fake")

    parser.add_argument("--model_trunk_num_filters", nargs="*", type=int,
        default=[64, 128, 196, 128, 128, 196, 128, 128, 196, 128]
    )
    parser.add_argument("--model_head_num_filters", nargs="*", type=int, default=[128, 64])
    parser.add_argument("--model_pool_layer_indices", nargs="*", type=int, default=[3, 6, 9])
    
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--savedir", type=str, default="/tmp/path/to/save_dir/")
    parser.add_argument("--seed", type=int, default=123)



def run_benchmark(model_file, img_loader):
    elapsed = 0
    model = torch.jit.load(model_file)
    model.eval()
    num_batches = 5
    # Run the scripted model on a few batches of images
    for i, (images, target) in enumerate(img_loader):
        if i < num_batches:
            start = time.time()
            output = model(images)
            end = time.time()
            elapsed = elapsed + (end-start)
        else:
            break
    num_images = images.size()[0] * num_batches

    return 'Elapsed time: %3.0f ms' % (elapsed/num_images*1000)


def size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    # print('Size (MB):', os.path.getsize("temp.p")/1e6)
    size =  os.path.getsize("temp.p")/1e6
    os.remove('temp.p')
    return size


def train(args):

    scripted_float_model_file = "original_model.pth"
    scripted_quantized_model_file = "quantized_model.pth"

    set_seed(args.seed)

    os.makedirs(args.savedir, exist_ok=True)
    logger = create_logger("%s/logfile.log" % args.savedir, args)
    
    model = CustomCNN(args)
    logger.info(model)
    init_size = size_of_model(model)

    torch.jit.save(torch.jit.script(model), os.path.join(args.savedir, scripted_float_model_file))

    model.eval()
    model.fuse_model()
    model.qconfig = torch.quantization.default_qconfig
    logger.info(model.qconfig)

    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)
    logger.info(model)
    final_size = size_of_model(model)

    torch.jit.save(torch.jit.script(model), os.path.join(args.savedir, scripted_quantized_model_file))

    logger.info(f'Size (MB): {init_size}')
    logger.info(f'Size (MB): {final_size}')

    train_dataset = LivenessDataset(args)

    data_loader_test = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers 
    )

    logger.info(
        run_benchmark(os.path.join(args.savedir, scripted_float_model_file), data_loader_test)
    )
    logger.info(
        run_benchmark(os.path.join(args.savedir, scripted_quantized_model_file), data_loader_test)
    )


def cli_main():
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    train(args)


if __name__ == '__main__':
    cli_main()
