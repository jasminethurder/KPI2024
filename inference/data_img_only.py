import os
import random
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import torch

class CustomDataset(Dataset):
    def __init__(self, root_dirs, mode='train', image_transform=None):
        self.root_dirs = root_dirs if isinstance(root_dirs, list) else [root_dirs]
        self.mode = mode
        self.image_transform = image_transform
        self.img_paths = []

        for root_dir in self.root_dirs:
            img_files = sorted(os.listdir(root_dir))
            for img_file in img_files:
                if img_file.lower().endswith('.png'):
                    img_path = os.path.join(root_dir, img_file)
                    self.img_paths.append(img_path)
                                        
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
                
        if self.image_transform:
            image = self.image_transform(image)

        if self.mode == 'train' or self.mode == 'trainval':
            image = self.apply_transforms(image)
        
        return image

    def apply_transforms(self, image):
        if random.random() < 0.3:  # 30% probability of rotation
            angle = transforms.RandomRotation.get_params([-45, 45])
            image = TF.rotate(image, angle)

        crop_ratio = random.uniform(0.8, 1)

        if random.random() < 0.3:  # 30% probability of random crop
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(int(512*crop_ratio), int(512*crop_ratio)))
            image = TF.crop(image, i, j, h, w)
            image = TF.resize(image, (512, 512))

        if random.random() < 0.3:  # 30% probability of horizontal flip
            image = TF.hflip(image)

        if random.random() < 0.3:  # 30% probability of vertical flip
            image = TF.vflip(image)

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

        return image

# Example usage:
if __name__ == "__main__":
    root_dirs = ['path_to_your_data_directory']
    dataset = CustomDataset(root_dirs, mode='train', image_transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for images in dataloader:
        print(images.shape)
