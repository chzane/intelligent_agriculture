import os
import random
import shutil

random.seed(42)

image_dir = '/Volumes/Zane/智慧农业/datasetsold/images'
label_dir = '/Volumes/Zane/智慧农业/datasetsold/labels'

train_image_dir = '/Volumes/Zane/智慧农业/datasets/train/images'
val_image_dir = '/Volumes/Zane/智慧农业/datasets/val/images'
test_image_dir = '/Volumes/Zane/智慧农业/datasets/test/images'

train_label_dir = '/Volumes/Zane/智慧农业/datasets/train/labels'
val_label_dir = '/Volumes/Zane/智慧农业/datasets/val/labels'
test_label_dir = '/Volumes/Zane/智慧农业/datasets/test/labels'

os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(test_image_dir, exist_ok=True)

os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)
os.makedirs(test_label_dir, exist_ok=True)

image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

random.shuffle(image_files)

train_split = 0.7
val_split = 0.2
test_split = 0.1

num_images = len(image_files)
train_size = int(train_split * num_images)
val_size = int(val_split * num_images)
test_size = num_images - train_size - val_size

train_files = image_files[:train_size]
val_files = image_files[train_size:train_size + val_size]
test_files = image_files[train_size + val_size:]

def copy_files(file_list, src_dir, dst_dir):
    for file_name in file_list:
        src_file = os.path.join(src_dir, file_name)
        dst_file = os.path.join(dst_dir, file_name)
        if os.path.exists(src_file):
            shutil.copy(src_file, dst_file)
        else:
            print(f"文件 {src_file} 不存在，跳过")

copy_files(train_files, image_dir, train_image_dir)
copy_files(val_files, image_dir, val_image_dir)
copy_files(test_files, image_dir, test_image_dir)

def get_label_file(image_file):
    return os.path.splitext(image_file)[0] + '.txt'

train_labels = [get_label_file(f) for f in train_files]
val_labels = [get_label_file(f) for f in val_files]
test_labels = [get_label_file(f) for f in test_files]

copy_files(train_labels, label_dir, train_label_dir)
copy_files(val_labels, label_dir, val_label_dir)
copy_files(test_labels, label_dir, test_label_dir)

