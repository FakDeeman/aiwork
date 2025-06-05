import os
import shutil
from sklearn.model_selection import train_test_split

# 配置路径
raw_root = "plantvillage_raw"  # 原始数据根目录
color_dir = os.path.join(raw_root, "color")  # 原始数据具体路径
output_root = "plantvillage_processed"  # 输出目录
test_ratio = 0.2  # 测试集比例

# 创建输出目录
os.makedirs(os.path.join(output_root, "train"), exist_ok=True)
os.makedirs(os.path.join(output_root, "test"), exist_ok=True)

# 遍历每个类别目录（例如 Apple__Apple_scab）
for class_name in os.listdir(color_dir):
    class_dir = os.path.join(color_dir, class_name)

    # 跳过非目录文件（如.DS_Store）
    if not os.path.isdir(class_dir):
        continue

    # 获取所有图片文件（递归子目录，但你的数据似乎没有嵌套）
    image_paths = []
    for root, _, files in os.walk(class_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))

    # 跳过空类别
    if not image_paths:
        print(f"警告: 类别 {class_name} 无图片，已跳过")
        continue

    # 分割数据集
    train_files, test_files = train_test_split(
        image_paths,
        test_size=test_ratio,
        random_state=42,
        shuffle=True
    )

    # 创建目标目录
    train_target_dir = os.path.join(output_root, "train", class_name)
    test_target_dir = os.path.join(output_root, "test", class_name)
    os.makedirs(train_target_dir, exist_ok=True)
    os.makedirs(test_target_dir, exist_ok=True)

    # 复制文件（含异常处理）
    for file_list, target_dir in [(train_files, train_target_dir), (test_files, test_target_dir)]:
        for src_path in file_list:
            try:
                shutil.copy(src_path, target_dir)
            except PermissionError:
                print(f"权限拒绝: 无法复制 {src_path}，请关闭占用文件的程序")
            except Exception as e:
                print(f"未知错误: {e}")

print(f"数据集分割完成！输出目录结构:\n{output_root}")
print(f"训练集: {len(train_files)} 张/类\n测试集: {len(test_files)} 张/类")