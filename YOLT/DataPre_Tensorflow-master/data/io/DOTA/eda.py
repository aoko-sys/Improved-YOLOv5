# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
# @Time    : 2021/4/2 13:54
# @Author  : DYQ
"""
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from bokeh.plotting import figure
from bokeh.io import output_notebook, show, output_file
from bokeh.models import ColumnDataSource, HoverTool, Panel
from bokeh.models.widgets import Tabs
import albumentations as albu
import glob

# ----------训练集和测试集的数量（1）------------
# Setup the paths to train and test images
# 训练图片路径
TRAIN_DIR = 'D:/YOLT/TianChi/project/tcdata/tile_round1_train_20201231/images/'

# 测试图片路径
TEST_DIR = 'tile_round1_testA_20201231/testA_imgs/'

# xml文件
TRAIN_CSV_PATH = 'D:/YOLT\TianChi/project/tcdata/tile_round1_train_20201231/instances_train2014.json'

# Glob the directories and get the lists of train and test images
train_fns = glob.glob(TRAIN_DIR + "/*.jpg")
test_fns = glob.glob(TEST_DIR + "/*.jpg")
print('Number of train images is {}'.format(len(train_fns)))
print('Number of test images is {}'.format(len(test_fns)))

# ----------每张图BBox的数量（2）------------
# # (1)读取json文件转换成dataframe
import json
train = []
with open(TRAIN_CSV_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)
train = pd.DataFrame(data)  # (15230, 5)
print(train.shape)

# # (2)按照name进行合并，观察bbox
all_train_images = pd.DataFrame(fns.split('\\')[-1] for fns in train_fns)
all_train_images.columns = ['name']
# merge image with json info
all_train_images = all_train_images.merge(train, on='name', how='left')

# replace nan values with zeros
all_train_images['bbox'] = all_train_images.bbox.fillna('[0,0,0,0]')
all_train_images.head(5)

# # (3)拆分bbox坐标方便后续观察
# [xmin, ymin, xmax, ymax]
bbox_items = all_train_images.bbox
all_train_images['bbox_xmin'] = bbox_items.apply(lambda x: x[0])
all_train_images['bbox_ymin'] = bbox_items.apply(lambda x: x[1])
all_train_images['bbox_width'] = bbox_items.apply(lambda x: x[2] - x[0])
all_train_images['bbox_height'] = bbox_items.apply(lambda x: x[3] - x[1])

# print(all_train_images)
print('{} images without bbox.'.format(len(all_train_images) - len(train)))


def get_all_bboxes(df, name):
    image_bboxes = df[df.name == name]

    bboxes = []
    for _, row in image_bboxes.iterrows():
        bboxes.append((row.bbox_xmin, row.bbox_ymin, row.bbox_width, row.bbox_height))

    return bboxes


def plot_image_examples(df, rows=3, cols=3, title='Image examples'):
    fig, axs = plt.subplots(rows, cols, figsize=(10, 10))
    for row in range(rows):
        for col in range(cols):
            idx = np.random.randint(len(df), size=1)[0]
            name = df.iloc[idx]["name"]
            img = Image.open(TRAIN_DIR + str(name))
            axs[row, col].imshow(img)

            bboxes = get_all_bboxes(df, name)

            for bbox in bboxes:
                rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r',
                                         facecolor='none')
                axs[row, col].add_patch(rect)

            axs[row, col].axis('off')

    plt.suptitle(title)


if __name__ == '__main__':
    plot_image_examples(all_train_images)