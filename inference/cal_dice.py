import numpy as np
import tifffile as tiff
import argparse
from PIL import Image

def resize_and_save(image, size, save_path):
    img = Image.fromarray(image)
    img = img.resize(size, Image.LANCZOS)
    img.save(save_path)

def compute_dice_coefficient(gt_path, pred_path, resize_size=(4096, 4096)):
    # Load the ground truth and prediction images
    gt = tiff.imread(gt_path)
    pred = tiff.imread(pred_path)
    
    # print('Original gt shape:', gt.shape)
    # print('Original pred shape:', pred.shape)
    
    # Resize images and save as PNG
    resize_and_save(gt, resize_size, 'gt_resized.png')
    resize_and_save(pred, resize_size, 'pred_resized.png')
    
    # Ensure the images are binary
    gt = (gt > 0).astype(np.uint8)
    pred = (pred > 0).astype(np.uint8)
    
    # print('Sum of pred:', np.sum(pred))
    
    # Compute the Dice coefficient
    intersection = np.sum(gt * pred)
    gt_sum = np.sum(gt)
    pred_sum = np.sum(pred)
    
    dice_coefficient = (2.0 * intersection) / (gt_sum + pred_sum)
    
    return dice_coefficient

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Dice coefficient between two images.")
    parser.add_argument('gt_path', type=str, help="Path to the ground truth image")
    parser.add_argument('pred_path', type=str, help="Path to the predicted image")
    
    args = parser.parse_args()
    
    dice = compute_dice_coefficient(args.gt_path, args.pred_path)
    print(f"Dice Coefficient: {dice}")
