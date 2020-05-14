import os
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


def get_transforms(args):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class LivenessDataset(Dataset):

	def __init__(self, args):
		self.pos_files = [os.path.join(args.pos_path, f) for f in os.listdir(args.pos_path)]
		self.neg_files = [os.path.join(args.neg_path, f) for f in os.listdir(args.neg_path)]

		self.img_transforms = get_transforms(args)


	def __len__(self):
		return len(self.pos_files) + len(self.neg_files)


	def __getitem__(self, idx):
		if idx < len(self.pos_files):
			path = self.pos_files[idx]
			label = torch.ones((32, 32))
		else:
			path = self.neg_files[idx - len(self.pos_files)]
			label = torch.zeros((32, 32))

		img = Image.open(path).convert("RGB")
		img = self.img_transforms(img)
		return img, label
