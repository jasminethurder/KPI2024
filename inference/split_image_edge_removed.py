import os
from PIL import Image
import json
import tifffile as tiff
import numpy as np

# Increase the maximum image size limit to avoid DecompressionBombError
Image.MAX_IMAGE_PIXELS = None

def split_image(image_path, output_dir, window_size=2048, overlap=1024):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the image using tifffile
    img = tiff.imread(image_path)
    print('Image already opened!')
    img_height, img_width = img.shape[:2]
    print(f'image shape: {img.shape}')
    if img_height * img_width > 2e9:
        window_size = int(2048)
        overlap = 1024
    else:
        window_size = int(2048)
        overlap = 1024
    print(f'window_size: {window_size}, overlap: {overlap}')
    
    # Initialize the dictionary to save coordinates and original image size
    coord_dict = {'original_size': (img_width, img_height)}
    coord_dict['window_size'] = window_size
    coord_dict['overlap'] = overlap
    
    count = 0
    total_steps = ((img_height + window_size - overlap - 1) // (window_size - overlap) + 1) * ((img_width + window_size - overlap - 1) // (window_size - overlap) + 1)
    for y in range(0, img_height, window_size - overlap):
        for x in range(0, img_width, window_size - overlap):
            y1 = y
            y2 = min(y + window_size, img_height)
            x1 = x
            x2 = min(x + window_size, img_width)
            # Crop the image
            cropped_img = img[y1:y2, x1:x2]
            
            # Create a new image of window_size x window_size with black background
            padded_img = np.zeros((window_size, window_size, img.shape[2]), dtype=img.dtype)
            padded_img[:(y2-y1), :(x2-x1)] = cropped_img
            
            # Save the cropped image using tifffile
            tiff.imwrite(os.path.join(output_dir, f'case_{count:05d}.png'), padded_img)
            # Store the coordinates and original size in the dictionary
            coord_dict[f'case_{count:05d}'] = {
                'top_left': (x, y),
                'size': (x2 - x1, y2 - y1)
            }
            count += 1
            print(f'\rProcessing: {count}/{total_steps} images', end='')

    # Save the dictionary to a JSON file
    with open(os.path.join(output_dir, 'coordinates.json'), 'w') as f:
        json.dump(coord_dict, f)

    print(f"\nTotal {count} images saved.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python split_image.py <image_path> <output_dir>")
    else:
        image_path = sys.argv[1]
        output_dir = sys.argv[2]
        split_image(image_path, output_dir)
