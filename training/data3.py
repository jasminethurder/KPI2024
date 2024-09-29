import os
import random
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np

import torch

class CustomDataset(Dataset):
    def __init__(self, root_dirs, mode='train', image_transform=None, mask_transform=None):
        self.root_dirs = root_dirs if isinstance(root_dirs, list) else [root_dirs]
        self.mode = mode
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.img_paths = []
        self.mask_paths = []

        for root_dir in self.root_dirs:
            for channel in os.listdir(root_dir):
                channel_path = os.path.join(root_dir, channel)
                if os.path.isdir(channel_path):
                    for case in os.listdir(channel_path):
                        case_path = os.path.join(channel_path, case)
                        # print(case_path)
                        if os.path.isdir(case_path):
                            img_folder = os.path.join(case_path, 'img')
                            mask_folder = os.path.join(case_path, 'mask')
                            # print(img_folder)
                            if os.path.exists(img_folder) and os.path.exists(mask_folder):
                                img_files = sorted(os.listdir(img_folder))
                                for img_file in img_files:
                                    img_path = os.path.join(img_folder, img_file)
                                    mask_file = img_file.replace('_img', '_mask')
                                    mask_path = os.path.join(mask_folder, mask_file)
                                    if os.path.exists(mask_path):
                                        self.img_paths.append(img_path)
                                        self.mask_paths.append(mask_path)
                                        
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Assuming masks are single channel (grayscale)
                
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        if self.mode == 'train' or self.mode == 'trainval':
            image, mask = self.apply_transforms(image, mask)
        
        return image, mask

    def apply_transforms(self, image, mask):
        if random.random() < 0.3:  # 30% probability of rotation
            angle = transforms.RandomRotation.get_params([-45, 45])
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)   

        crop_ratio = random.uniform(0.8, 1)

        if random.random() < 0.3:  # 30% probability of random crop
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(int(512*crop_ratio), int(512*crop_ratio)))
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)
            image = TF.resize(image, (512, 512))
            mask = TF.resize(mask, (512, 512))

        if random.random() < 0.3:  # 30% probability of horizontal flip
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() < 0.3:  # 30% probability of vertical flip
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        if random.random() < 0.3:  # 30% probability of color jitter
            image = TF.to_pil_image(image)
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))
            image = TF.to_tensor(image)

        return image, mask
