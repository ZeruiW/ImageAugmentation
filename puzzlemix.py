import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import io
import json
import base64
import mixup
from mixup import mixup_graph

def puzzlemix(image1, image2):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    mean_torch = mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    std_torch = std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    resnet = models.resnet18(pretrained=True)
    resnet.eval()

    
    # Define the image transformation pipeline
    transform_pipeline = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Transform the input images
    input_image1 = transform_pipeline(image1)
    input_image2 = transform_pipeline(image2)

    # Add an extra dimension to represent the batch size (which is 1 in this case)
    input_image1 = input_image1.unsqueeze(0)
    input_image2 = input_image2.unsqueeze(0)


    with torch.no_grad():
        output1 = resnet(input_image1)
        output2 = resnet(input_image2)

    # Get the labels by finding the indices with the highest predicted probabilities
    _, label1 = torch.max(output1, 1)
    _, label2 = torch.max(output2, 1)

    # Convert labels to integers
    label1 = label1.item()
    label2 = label2.item()

    criterion = nn.CrossEntropyLoss()

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])

    tensor1 = test_transform(image1)
    tensor2 = test_transform(image2)

    input_sp = torch.stack([tensor1, tensor2], dim=0)
    # You need to provide appropriate labels for the input images
    targets = torch.tensor([label1, label2]) 

    input_var = input_sp.clone().detach().requires_grad_(True)
    output = resnet(input_var)
    loss = criterion(output, targets)
    loss.backward()

    unary = torch.sqrt(torch.mean(input_var.grad **2, dim=1))
    unary = unary / unary.view(2, -1).max(1)[0].view(2, 1, 1)

    unary16 = F.avg_pool2d(unary, 224//16)
    unary16 = unary16 / unary16.view(2, -1).max(1)[0].view(2, 1, 1)

    indices = [1, 0]
    n_labels = 3
    block_num = 16
    alpha = 0.4
    beta = 0.2
    gamma = 1.0
    eta = 0.2
    transport = True
    t_eps = 0.2
    t_size = 224 // block_num
    #parameter
    output = mixup_graph(input_sp, unary, indices=indices, n_labels=n_labels,
                        block_num=block_num, alpha=np.array([alpha]).astype('float32'), beta=beta, gamma=gamma, eta=eta,
                        neigh_size=2, mean=mean_torch, std=std_torch,
                        transport=transport, t_eps=t_eps, t_size=t_size,
                        device='cpu')

    input_me = output[0] * std_torch + mean_torch
    np_array = input_me.numpy()
    ratio = output[1][0]
    shuffled_targets = ratio * targets[0] + targets[1] * (1 - ratio)
    shuffled_targets = shuffled_targets.numpy()

    return np_array[0], shuffled_targets


from PIL import Image
import numpy as np

def save_image(np_array, file_name):
    print("Shape of mixed_image:", np_array.shape)  # Add this line to print the shape
    np_array = np_array.transpose((1, 2, 0))
    np_array = (np_array * 255).astype(np.uint8)
    img = Image.fromarray(np_array)
    img.save(file_name)


# Read the local image files
image1 = Image.open("ILSVRC2012_val_00021015.JPEG")
image2 = Image.open("ILSVRC2012_val_00028673.JPEG")

# Get the mixed image and the corresponding label
mixed_image, mixed_label = puzzlemix(image1, image2)

# Save the mixed image to a local file
save_image(mixed_image, "output.jpg")

# Print the mixed label
print("Mixed label:", mixed_label)


import os
import random
from PIL import Image

from tqdm import tqdm

def process_images(train_dir, alldata_dir, puzzlemixed_dir):
    for label in os.listdir(train_dir):
        label_dir = os.path.join(train_dir, label)
        puzzlemixed_label_dir = os.path.join(puzzlemixed_dir, label)

        if not os.path.exists(puzzlemixed_label_dir):
            os.makedirs(puzzlemixed_label_dir)

        all_images = [os.path.join(alldata_dir, img) for img in os.listdir(alldata_dir)]

        # 使用tqdm显示进度条
        for img_path in tqdm(os.listdir(label_dir), desc=f'Processing {label}'):
            img1_path = os.path.join(label_dir, img_path)
            image1 = Image.open(img1_path).convert("RGB")

            random_img_path = random.choice(all_images)
            image2 = Image.open(random_img_path).convert("RGB")

            mixed_image, mixed_label = puzzlemix(image1, image2)

            puzzlemixed_img_path = os.path.join(puzzlemixed_label_dir, img_path)
            save_image(mixed_image, puzzlemixed_img_path)




train_dir = "Train50_400"
alldata_dir = "Train50_1300selexted200"
puzzlemixed_dir = "puzzlemixed_Train50_400"

process_images(train_dir, alldata_dir, puzzlemixed_dir)
