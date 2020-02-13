# 从训练集随机挑出5000张猫狗图片（猫狗各一半）作为验证集
import os
import shutil
import random

base_dir = 'dogs-vs-cats-redux-kernels-edition'
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'valid')
index_list = random.sample(range(12500), 2500)
for index in index_list:
    source_file = "cat.{}.jpg".format(index)
    source_path = os.path.join(train_dir, source_file)
    target_path = os.path.join(valid_dir, source_file)
    if os.path.exists(source_path):
        shutil.move(source_path, target_path)

index_list = random.sample(range(12500), 2500)
for index in index_list:
    source_file = "dog.{}.jpg".format(index)
    source_path = os.path.join(train_dir, source_file)
    target_path = os.path.join(valid_dir, source_file)
    if os.path.exists(source_path):
        shutil.move(source_path, target_path)
