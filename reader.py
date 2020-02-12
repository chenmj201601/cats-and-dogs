import os
import cv2
import random
import numpy as np
from image_utils import transform_img


# 数据读取器
def data_loader(data_dir, batch_size=20, mode='train'):
    if mode == 'train':
        cats_dir = os.path.join(data_dir, 'Cat')
        dogs_dir = os.path.join(data_dir, 'Dog')
    else:
        cats_dir = os.path.join(data_dir, 'cat')
        dogs_dir = os.path.join(data_dir, 'dog')
    cat_names = os.listdir(cats_dir)
    dog_names = os.listdir(dogs_dir)
    if mode == 'train':
        cat_names = ['Cat/' + name for name in cat_names]
        dog_names = ['Dog/' + name for name in dog_names]
    else:
        cat_names = ['cat/' + name for name in cat_names]
        dog_names = ['dog/' + name for name in dog_names]
    file_names = cat_names + dog_names

    def reader():
        if mode == 'train':
            random.shuffle(file_names)
        batch_imgs = []
        batch_labels = []
        for name in file_names:
            file_path = os.path.join(data_dir, name)
            img = cv2.imread(file_path)
            img = transform_img(img)
            if name.startswith('Cat') or name.startswith('cat'):
                label = 0
            elif name.startswith('Dog') or name.startswith('dog'):
                label = 1
            else:
                raise ('Not excepted file name')
            if img.shape[0] == 4:
                print(file_path)
                print(img.shape)
                break
            batch_imgs.append(img)
            batch_labels.append(label)
            if len(batch_imgs) == batch_size:
                img_array = np.array(batch_imgs).astype('float32')
                labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
                yield img_array, labels_array
                batch_imgs = []
                batch_labels = []

        if len(batch_imgs) > 0:
            img_array = np.array(batch_imgs).astype('float32')
            labels_array = np.array(batch_labels).astype('float32').reshape(-1, 1)
            yield img_array, labels_array

    return reader


if __name__ == '__main__':
    data_dir = 'data/train'
    train_loader = data_loader(data_dir)
    for batch_id, data in enumerate(train_loader()):
        train_img, train_label = data
        print(train_img.shape)
        print(train_label.shape)
