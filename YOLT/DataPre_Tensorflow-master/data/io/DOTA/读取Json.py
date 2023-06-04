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

# xml文件
TRAIN_CSV_PATH = 'D:/YOLT/R3Det_Tensorflow-master/data/io/DOTA/instances_train2014.json'

# # (1)读取json文件转换成dataframe
import json
train = []
with open(TRAIN_CSV_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)
train = pd.DataFrame(data)  # (15230, 5)
print(train.shape)