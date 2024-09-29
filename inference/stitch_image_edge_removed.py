import os
from PIL import Image
import json
import numpy as np
import tifffile as tiff

def stitch_image(coord_dict_path, input_dir, output_image_path, window_size=4096, overlap=512):
    # Load the coordinates dictionary
    with open(coord_dict_path, 'r') as f:
        coord_dict = json.load(f)

    # Get the size of the original image
    original_size = coord_dict.pop('original_size')
    max_x, max_y = original_size
    window_size = coord_dict.pop('window_size')
    overlap = coord_dict.pop('overlap')
    print(f'window_size: {window_size}, overlap: {overlap}')

    # Create a blank image with the size of the original image
    stitched_img = Image.new('L', (max_x, max_y))  # 'L' mode for single channel (grayscale)
    
    # Initialize progress variables
    total_images = len(coord_dict)
    processed_images = 0

    # Paste each small image back to the original location
    for key, value in coord_dict.items():
        img_path = os.path.join(input_dir, f"{key}.png")
        img = Image.open(img_path).convert('L')  # Convert to single channel
        top_left = value['top_left']
        x, y = top_left

        # Calculate the center crop area (the middle region)
        center_crop_area = (
            overlap // 2,
            overlap // 2,
            window_size - overlap // 2,
            window_size - overlap // 2
        )

        # Crop the center region from the image
        center_crop = img.crop(center_crop_area)
        
        # Calculate the position to paste the center crop
        x_center = x + overlap // 2
        y_center = y + overlap // 2

        # Paste the center crop into the stitched image
        stitched_img.paste(center_crop, (x_center, y_center))
        
        # Update progress
        processed_images += 1
        print(f'\rProcessing: {processed_images}/{total_images} images', end='')

    # Convert the stitched image to numpy array
    stitched_img_np = np.array(stitched_img)

    # Save the stitched image as a TIFF file
    tiff.imwrite(output_image_path, stitched_img_np, dtype=np.uint8)
    print(f"\nStitched image saved to {output_image_path}")

    # Resize stitched image to 4096x4096
    resized_img = stitched_img.resize((4096, 4096), Image.LANCZOS)

    # Save the resized image as a PNG file
    png_output_path = os.path.splitext(output_image_path)[0] + ".png"
    resized_img.save(png_output_path, "PNG")
    print(f"Resized image saved to {png_output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python stitch_image.py <coord_dict_path> <input_dir> <output_image_path>")
    else:
        coord_dict_path = sys.argv[1]
        input_dir = sys.argv[2]
        output_image_path = sys.argv[3]
        print(output_image_path)
        stitch_image(coord_dict_path, input_dir, output_image_path)
