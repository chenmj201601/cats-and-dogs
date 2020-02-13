import os
import cv2
import random
import numpy as np
from image_utils import transform_img


# 数据读取器
def data_loader(data_dir, batch_size=20, mode='train'):
    file_names = os.listdir(data_dir)

    def reader():
        if mode == 'train':
            random.shuffle(file_names)
        batch_imgs = []
        batch_labels = []
        for name in file_names:
            file_path = os.path.join(data_dir, name)
            img = cv2.imread(file_path)
            img = transform_img(img)
            if name.startswith('cat'):
                label = 0
            elif name.startswith('dog'):
                label = 1
            else:
                raise ('Not excepted file name')
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
    data_dir = 'dogs-vs-cats-redux-kernels-edition/train'
    train_loader = data_loader(data_dir)
    for batch_id, data in enumerate(train_loader()):
        train_img, train_label = data
        print(train_img.shape)
        print(train_label.shape)
