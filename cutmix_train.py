import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import pickle
import mixup
from mixup import mixup_graph

def cut(image1, image2):
    resnet = models.resnet18(pretrained=True)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    mean_torch = mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    std_torch = std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(254),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])

    tensor1 = test_transform(image1)
    tensor2 = test_transform(image2)

    input_sp = torch.stack([tensor1, tensor2], dim=0)

    alpha = 1
    data = input_sp
    _, _, height, width = data.shape
    indices = [1, 0]
    shuffled_data = data[indices]

    lam = np.random.beta(alpha, alpha)

    image_h, image_w = data.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]

    image_final = (data * std_torch + mean_torch)[0]
    cutmix_image = Image.fromarray((image_final.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    
    return cutmix_image


import os
import random
from PIL import Image

def process_images(train_dir, alldata_dir, cutmixed_dir):
    for label in os.listdir(train_dir):
        label_dir = os.path.join(train_dir, label)
        cutmixed_label_dir = os.path.join(cutmixed_dir, label)

        if not os.path.exists(cutmixed_label_dir):
            os.makedirs(cutmixed_label_dir)

        all_images = [os.path.join(alldata_dir, img) for img in os.listdir(alldata_dir)]

        for img_path in os.listdir(label_dir):
            img1_path = os.path.join(label_dir, img_path)
            image1 = Image.open(img1_path).convert("RGB")

            random_img_path = random.choice(all_images)
            image2 = Image.open(random_img_path).convert("RGB")

            cutmix_image = cut(image1, image2)

            cutmix_img_path = os.path.join(cutmixed_label_dir, img_path)
            cutmix_image.save(cutmix_img_path)


train_dir = "Train50_200"
alldata_dir = "Train50_1300selexted200"
cutmixed_dir = "cutmixed"

process_images(train_dir, alldata_dir, cutmixed_dir)
