import os
import numpy as np
import nibabel as nib
from PIL import Image
import json

# 定义输入目录和字典文件路径

input_dir = 'img_inference_gz7'
# input_dir = 'image_gz/Dataset111_patch_kidney/imagesTr'
# input_dir = '/home/cvailab/nnUNet/ensemble_results/ensemble_2d98.05_3d_fullres96.03_3d_lowres94.85_ResEncUNetM_3d_lowres98.49'  
dict_path = 'image_gz/modified_data.json'  

# 加载保存的字典
with open(dict_path, 'r') as f:
    file_path_dict = json.load(f)

# 列出输入目录中的所有文件
for file_name in os.listdir(input_dir):
    # print(file_name)
    if file_name.endswith('.nii.gz') and file_name in file_path_dict:
        # 构建完整的文件路径
        file_path = os.path.join(input_dir, file_name)
        
        # 加载/home/cvailab/nnUNet/ensemble_results/xlstm_3d_lowres95.34NIfTI文件
        nifti_img = nib.load(file_path)
        
        # 将NIfTI图像转换为numpy数组
        img_array = nifti_img.get_fdata()
        
        # 如果图像是3D的，只取第一个切片
        if img_array.ndim == 3:
            img_array = img_array[:, :, 0]
        
        # 检查无效值并处理
        if np.any(np.isnan(img_array)) or np.any(np.isinf(img_array)):
            print(f'Warning: Invalid values found in {file_name}. Replacing with zeros.')
            img_array = np.nan_to_num(img_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 归一化图像数据到0-255范围
        if np.max(img_array) != np.min(img_array):  # 防止除以零
            img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array)) * 255.0
        img_array = img_array.astype(np.uint8)
        
        # 转换numpy数组为图像
        img = Image.fromarray(img_array)
        
        # 获取输出文件路径
        output_file_path = file_path_dict[file_name]
        
        # 创建不存在的文件夹路径
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        
        # 保存图像为png文件
        # print(output_file_path)
        img.save(output_file_path)
        # print(f'Processed and saved {file_path} to {output_file_path}')

print("Conversion complete!")