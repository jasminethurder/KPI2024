import torch
import os
import argparse
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm
from data_img_only import CustomDataset  # Assuming you have saved your dataset class as custom_dataset.py

def remove_module_prefix(state_dict):
    """
    Remove 'module.' prefix from the keys in state_dict.
    """
    new_state_dict = {}
    for key in state_dict:
        new_key = key.replace("module.", "")
        new_state_dict[new_key] = state_dict[key]
    return new_state_dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = smp.UnetPlusPlus(encoder_name='timm-regnety_120', 
#                          encoder_weights=None, 
#                          in_channels=3, classes=2)
# model.load_state_dict(remove_module_prefix(torch.load("/home/cvailab/task2_docker/src/checkpoints/timm-regnety_120_0729.pth", weights_only=True)))
# model.to(device)
# model.eval()

# model2 = smp.UnetPlusPlus(encoder_name='timm-regnety_080', 
#                          encoder_weights=None, 
#                           in_channels=3, classes=2)
# model2.load_state_dict(remove_module_prefix(torch.load("/home/cvailab/task2_docker/src/checkpoints/timm-regnety_080_0727.pth", weights_only=True)))
# model2.to(device)
# model2.eval()

# model3 = smp.UnetPlusPlus(encoder_name='efficientnet-b3', 
#                          encoder_weights=None, 
#                           in_channels=3, classes=2)
# model3.load_state_dict(remove_module_prefix(torch.load("/home/cvailab/task2_docker/src/checkpoints/efficientnet-b3_0727.pth", weights_only=True)))
# model3.to(device)
# model3.eval()

# model4 = smp.DeepLabV3Plus(encoder_name='timm-regnety_120', 
#                          encoder_weights=None, 
#                           in_channels=3, classes=2)
# model4.load_state_dict(remove_module_prefix(torch.load("timm-regnety_120_0729_D.pth", weights_only=True)))
# model4.to(device)
# model4.eval()

model4 = smp.UnetPlusPlus(encoder_name='resnet50', 
                         encoder_weights=None, 
                          in_channels=3, classes=2)
model4.load_state_dict(remove_module_prefix(torch.load("resnet50Upp2.pth", weights_only=True)))
model4.to(device)
model4.eval()

# model5 = smp.UnetPlusPlus(encoder_name='timm-regnety_120', 
#                          encoder_weights=None, 
#                           in_channels=3, classes=2)
# model5.load_state_dict(remove_module_prefix(torch.load("/home/cvailab/task2_docker/src/checkpoints/timm-regnety_120_supp.pth", weights_only=True)))
# model5.to(device)
# model5.eval()

# model6 = smp.UnetPlusPlus(encoder_name='timm-regnety_016', 
#                          encoder_weights=None, 
#                           in_channels=3, classes=2)
# model6.load_state_dict(remove_module_prefix(torch.load("/home/cvailab/task2_docker/src/checkpoints/timm-regnety_016_supp.pth", weights_only=True)))
# model6.to(device)
# model6.eval()



@torch.no_grad()
def build_labels(inputs, *models):
    inputs = inputs.to(device)
    pred_masks = sum(model(inputs) for model in models)
    pred_masks /= len(models)
    pred_masks_final = pred_masks[:, 1, :, :] > pred_masks[:, 0, :, :]
    return pred_masks, pred_masks_final

image_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    # Add more transformations if needed
])

def main(input_dir, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    val_set = CustomDataset(root_dirs=input_dir,
                               mode='val',
                               image_transform=image_transform)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, drop_last=False)
    print(f'Length of val_set: {len(val_set)}')

    # Visualize predictions
    for i, images in enumerate(tqdm(val_loader, desc="Processing validation set")):
        images = images.to(device)

        with torch.no_grad():
            # _, pred_masks = build_labels(images, model, model2, model3, model4)
            _, pred_masks = build_labels(images, model4)

        pred_masks_resized = F.interpolate(pred_masks.unsqueeze(0).float(), size=(4096, 4096), mode='bilinear', align_corners=False).cpu()
        pred_masks_resized_np = pred_masks_resized.squeeze().cpu().numpy()

        # 将数值范围从 [0, 1] 转换为 [0, 255]
        pred_masks_resized_np = (pred_masks_resized_np * 255).astype(np.uint8)

        # 将 NumPy 数组转换为 PIL 图像
        img = Image.fromarray(pred_masks_resized_np)

        # 保存为 PNG 文件
        img.save(f'{output_dir}/case_{i:05d}.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some images.")
    parser.add_argument('input_dir', type=str, help='Directory of input images')
    parser.add_argument('output_dir', type=str, help='Directory to save output images')

    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
