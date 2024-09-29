#!/bin/bash

# 检查输入参数
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <image_path> <output_dir> <overlap>"
    exit 1
fi

# 获取输入参数
image_path=$1
output_dir=$2
overlap=$3

# 定义临时目录
temp_dir="Task2_split/12-173_wsi"

# 拆分图片
echo "Splitting image..."
python split_image_edge_removed.py "$image_path" "$temp_dir/img" "$overlap"

# 创建目标目录
echo "Creating mask directory..."
mkdir -p "$temp_dir/mask"

# 复制 coordinates.json 文件
echo "Copying coordinates.json..."
cp "$temp_dir/img/coordinates.json" "$temp_dir/mask/coordinates.json"

# 进行推理预测
echo "Running inference prediction..."
python inference_prediction.py "$temp_dir/img" "$temp_dir/mask"

# 拼接图片
echo "Stitching image..."
python stitch_image_edge_removed.py "$temp_dir/mask/coordinates.json" "$temp_dir/mask" "$output_dir"

# 删除临时目录
echo "Cleaning up..."
rm -rf "$temp_dir"

echo "Process completed."

# 将 image_path 中的 img 替换为 mask
mask_path=${image_path//wsi/mask}
echo "mask_path: $mask_path"

# 将 overlap 参数写入 dice.txt 文件
echo "Overlap: $overlap" >> dice.txt

# 计算 Dice 系数
python cal_dice.py "$mask_path" "$output_dir" >> dice.txt
