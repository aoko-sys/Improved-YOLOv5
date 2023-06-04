import os
import glob
import json
import shutil
import numpy as np
import xml.etree.ElementTree as ET


path2 = '.'  # 当前该文件路径

START_BOUNDING_BOX_ID = 1



classes = ['L_Y', 'JY1_2', 'Emergency', 'Handle', 'AR', 'QF_A', 'XT_13', 'XT_14', 'XT_15', 'FR_R']  # 类别
pre_define_categories = {}
for i, cls in enumerate(classes):
    pre_define_categories[cls] = i+1

only_care_pre_define_categories = True
# only_care_pre_define_categories = False

train_ratio = 1  # 控制train和val的比例 train_ratio=1是全部生成为train数据
save_json_train = 'instances_train2014.json'
save_json_val = 'instance_val2014.json'
xml_dir = "Annotation"   # 存放xml文件的文件夹

xml_list = glob.glob(xml_dir + "/*.xml")
xml_list = np.sort(xml_list)
np.random.seed(100)
np.random.shuffle(xml_list)

train_num = int(len(xml_list)*train_ratio)
xml_list_train = xml_list[:train_num]
xml_list_val = xml_list[train_num:]


xml_list = xml_list_train
json_file = save_json_train
