import argparse
import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import LivenessDataset
from helpers import *
from logger import create_logger
from model import CustomCNN


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


def train(args):

    set_seed(args.seed)

    os.makedirs(args.savedir, exist_ok=True)
    logger = create_logger("%s/logfile.log" % args.savedir, args)
    
    model = CustomCNN(args)
    logger.info(model)

    train_dataset = LivenessDataset(args)
    logger.info(f"Train Dataset Size: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers 
    )

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    criterion = nn.MSELoss()

    for i_epoch in tqdm(range(args.num_epochs)):
        
        model.train()
        for batch in tqdm(train_loader):
            images, labels = batch
            out = model(images).squeeze(1)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # Change to Validation Dataset 
        model.eval()
        for batch in train_loader:
            images, labels = batch
            with torch.no_grad():
                out = model(images).squeeze(1)
                loss = criterion(out, labels)


def cli_main():
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    train(args)


if __name__ == '__main__':
    cli_main()

