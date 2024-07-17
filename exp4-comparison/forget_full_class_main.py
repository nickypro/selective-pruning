#!/bin/python3.8

"""
This file is used to collect all arguments for the experiment, prepare the dataloaders, call the method for forgetting, and gather/log the metrics.
Methods are executed in the strategies file.
"""

import random
import os
import wandb

# import optuna
from typing import Tuple, List
import sys
import argparse
import time
from copy import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, dataset
from torch.utils.data import Subset
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

print("torch.cuda.is_available()", torch.cuda.is_available())

import models
from unlearn import *
from utils import *
import forget_full_class_strategies
import datasets
import models
import conf
from training_utils import *

# import differently from huggingface
from datasets import load_dataset
from cifar_classes import cifar_classes
from PIL import Image
from torchvision import transforms

print("--> End of imports")

"""
Get Args
"""
parser = argparse.ArgumentParser()
parser.add_argument("-net", type=str, required=True, help="net type")
parser.add_argument(
    "-weight_path",
    type=str,
    required=True,
    help="Path to model weights. If you need to train a new model use pretrain_model.py",
)
parser.add_argument(
    "-dataset",
    type=str,
    required=True,
    nargs="?",
    choices=["Cifar10", "Cifar20", "Cifar100", "PinsFaceRecognition"],
    help="dataset to train on",
)
parser.add_argument("-classes", type=int, required=True, help="number of classes")
parser.add_argument("-gpu", action="store_true", default=False, help="use gpu or not")
parser.add_argument("-b", type=int, default=64, help="batch size for dataloader")
parser.add_argument("-warm", type=int, default=1, help="warm up training phase")
parser.add_argument("-lr", type=float, default=0.1, help="initial learning rate")
parser.add_argument("-frac", type=float, default=0.03, help="fraction of neurons so selectively prune")
parser.add_argument("-state_dict_dir", type=str, default=None, help="choose modified base params to load")
parser.add_argument(
    "-method",
    type=str,
    required=True,
    nargs="?",
    choices=[
        "baseline",
        "retrain",
        "finetune",
        "blindspot",
        "amnesiac",
        "UNSIR",
        "NTK",
        "ssd_tuning",
        "FisherForgetting",
        "selective_pruning",
        "load_modified_base",
        "load_modified_final",
        "load_modified_both",
    ],
    help="select unlearning method from choice set",
)
parser.add_argument(
    "-forget_class",
    type=str,
    required=True,
    nargs="?",
    help="class to forget",
    choices=list(conf.class_dict),
)
parser.add_argument(
    "-epochs", type=int, default=1, help="number of epochs of unlearning method to use"
)
parser.add_argument("-seed", type=int, default=0, help="seed for runs")
args = parser.parse_args()

# Set seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


# Check that the correct things were loaded
if args.dataset == "Cifar20":
    assert args.forget_class in conf.cifar20_classes
elif args.dataset == "Cifar100":
    assert args.forget_class in conf.cifar100_classes

forget_class = conf.class_dict[args.forget_class]

batch_size = args.b


# get network
net = getattr(models, args.net)(num_classes=args.classes)
#net.load_state_dict(torch.load(args.weight_path))


# for bad teacher
unlearning_teacher = getattr(models, args.net)(num_classes=args.classes, random_init=True)

if args.gpu:
    net = net.cuda()
    unlearning_teacher = unlearning_teacher.cuda()

# For celebritiy faces
root = "105_classes_pins_dataset" if args.dataset == "PinsFaceRecognition" else "./data"

# Scale for ViT (faster training, better performance)
print("--> loading datasets")
img_size = 224 if args.net == "ViT" else 32

# Load CIFAR-100 dataset
dataset = load_dataset("cifar100", streaming=False)

# Define a transform to resize the images
#img_size = 224 if args.net == "ViT" else 32
#transform = transforms.Compose([
#    transforms.Resize((img_size, img_size)),
#    transforms.ToTensor()
#])

# Apply the transform to each image in the dataset
#def resize_images(example):
#    example['img'] = transform(example['img'])
#    return example


class LazyTransformDataset(Dataset):
    def __init__(self, hf_dataset, transform=None, indices=None):
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.indices = np.array(list(range(len(self.hf_dataset))) if indices is None else indices)


    def __len__(self):
        return len(self.indices)
        #return len(self.hf_dataset)

    def __getitem__(self, idx):
        sub_index = self.indices[idx]
        item = self.hf_dataset[int(sub_index)]
        if self.transform:
            item = self.transform(item)
        return item

    def filter(self, filter_func):
        return LazyTransformDataset(
            self.hf_dataset.filter(filter_func),
            self.transform,
        )

    def random_subset(self, frac=0.3):
        num_samples = int(frac * len(self))
        new_indices = np.array(random.sample(sorted(self.indices), num_samples))
        return LazyTransformDataset(
            self.hf_dataset,
            self.transform,
            new_indices,
        )


def process_img(example):
    example['img'] = torch.tensor(net.processor(example['img'])["pixel_values"][0])
    return example

#dataset = dataset.map(process_img)

trainset = LazyTransformDataset( dataset["train"], process_img)
validset = LazyTransformDataset( dataset["test"],  process_img)

#trainset = getattr(datasets, args.dataset)(
#    root=root, download=True, train=True, unlearning=True, img_size=img_size
#)
#validset = getattr(datasets, args.dataset)(
#    root=root, download=True, train=False, unlearning=True, img_size=img_size
#)

print("--> get forget retain subsets")
def split_dataset(dataset, label):
    def forget_filter(example):
        return example["fine_label"] == label
    def retain_filter(example):
        return example["fine_label"] != label

    forget_set = dataset.filter(forget_filter)
    retain_set = dataset.filter(retain_filter)

    return forget_set, retain_set

forget_valid, retain_valid = split_dataset(validset, forget_class)
forget_train, retain_train = split_dataset(trainset, forget_class)

print("--> combining into dataloader")
forget_valid_dl = DataLoader(forget_valid, batch_size)
retain_valid_dl = DataLoader(retain_valid, batch_size)
validloader = DataLoader(validset, batch_size)

forget_train_dl = DataLoader(forget_train, batch_size)
retain_train_dl = DataLoader(retain_train, batch_size)
#retain_train_dl = create_data_loader(retain_train, batch_size)

full_train_dl = DataLoader(trainset, batch_size)
#full_train_dl = DataLoader(
#    ConcatDataset((retain_train_dl.dataset, list(forget_train_dl.dataset))),
#    batch_size=batch_size,
#)

print("--> end of dataloading")

# Change alpha here as described in the paper
# For PinsFaceRe-cognition, we use α=50 and λ=0.1
model_size_scaler = 1
if args.net == "ViT":
    model_size_scaler = 0.5
else:
    model_size_scaler = 1

kwargs = {
    "model": net,
    "unlearning_teacher": unlearning_teacher,
    "retain_train_dl": retain_train_dl,
    "retain_valid_dl": retain_valid_dl,
    "forget_train_dl": forget_train_dl,
    "forget_valid_dl": forget_valid_dl,
    "full_train_dl": full_train_dl,
    "valid_dl": validloader,
    "dampening_constant": 1,
    "selection_weighting": 10 * model_size_scaler,
    "forget_class": forget_class,
    "num_classes": args.classes,
    "dataset_name": args.dataset,
    "device": "cuda" if args.gpu else "cpu",
    "model_name": args.net,
    "frac": args.frac,
    "state_dict_dir": args.state_dict_dir,
}

# Logging
wandb.init(
    project=f"R0_{args.net}_{args.dataset}_fullclass_{args.forget_class}",
    name=f"{args.method}",
)

# Time the method
import time

print("--> starting run")
start = time.time()


# executes the method passed via args
testacc, retainacc, zrf, mia, d_f = getattr(forget_full_class_strategies, args.method)(
    **kwargs
)
end = time.time()
time_elapsed = end - start
print(testacc, retainacc, zrf, mia)

# Logging
wandb.log(
    {
        "TestAcc": testacc,
        "RetainTestAcc": retainacc,
        "ZRF": zrf,
        "MIA": mia,
        "Df": d_f,
        "MethodTime": time_elapsed,  # do not forget to deduct baseline time from it to remove results calc (acc, MIA, ...)
    }
)
wandb.finish()
