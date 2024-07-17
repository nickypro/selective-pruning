"""
From https://github.com/vikram2000b/bad-teaching-unlearning
And https://github.com/weiaicunzai/pytorch-cifar100 (better performance) <- Refer to this for comments
"""

import copy
import itertools
from torch import nn
import numpy as np
import torch
from transformers import ViTModel, ViTFeatureExtractor, ViTForImageClassification, AutoImageProcessor
from taker import Model

class ViT():
    def __init__(self, num_classes=20, random_init:bool = False, **kwargs):
        if random_init:
            m = Model("nickypro/vit-cifar100-random-init", dtype="fp32")
        else:
            m = Model("nickypro/vit-cifar100", dtype="fp32")

        self.taker_model = m
        #AutoImageProcessor.from_pretrained("nickypro/vit-cifar100")
        self.processor = m.processor
        self.base  = m.predictor.vit
        self.final = m.predictor.classifier
        self.num_classes = num_classes
        self.device = next(self.base.parameters()).device
        print("DEVICE:", self.device)

    def cuda(self, **kwargs):
        self.taker_model.to("cuda")
        self.device = "cuda"
        return self

    def forward(self, pixel_values):
        #pixel_values = torch.tensor(np.array(
        #    self.processor(img["pixel_values"])
        #)).to(self.device)
        pixel_values = pixel_values.to(self.device)
        outputs = self.base(pixel_values=pixel_values)
        logits = self.final(outputs.last_hidden_state[:, 0])

        return logits

    def parameters(self):
        p = [
            self.base.parameters(),
            self.final.parameters(),
        ]
        return itertools.chain(*p)

    def named_parameters(self):
        p = [
            self.base.named_parameters(),
            self.final.named_parameters(),
        ]
        return itertools.chain(*p)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def eval(self):
        self.base = self.base.eval()
        self.final = self.final.eval()
        return self

    def train(self):
        self.base = self.base.train()
        self.final = self.final.train()
        return self

    def save(self, filename):
        state_dict = {
            **{f"base.{k}":  v.cpu() for k, v in self.base.state_dict().items()},
            **{f"final.{k}": v.cpu() for k, v in self.final.state_dict().items()},
        }
        torch.save(state_dict, filename)


    def __deepcopy__(self, memo):
        # Create a deep copy of the class without the tensor
        new_model = copy.copy(self)
        new_model.taker_model = copy.deepcopy(self.taker_model, memo)
        new_model.processor = new_model.taker_model.processor
        new_model.base      = new_model.taker_model.predictor.vit
        new_model.final     = new_model.taker_model.predictor.classifier
        new_model.device    = next(new_model.base.parameters()).device
        return new_model
