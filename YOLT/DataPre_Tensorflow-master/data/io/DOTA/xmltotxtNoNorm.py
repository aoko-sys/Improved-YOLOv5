'''
时间：2021-4-1

此处代码为，将图片对应的xml标注文件文件---->.txt标注文件
不对xmin  ymin   xmax  ymax 进行中心化 和 归一化 操作，只是换一种文件存储
'''


# 导包
import xml.etree.ElementTree as ET


# 类别列表（根据自己的开发需求的实际情况填写）
classes = ['Handle,', 'Emergency', 'JY1_2', 'L_Y']  # 类别


# label中锚框坐标归一化
def convert(size, box):  # size:(原图图像w,原图图像h) , box:(xmin, xmax, ymin, ymax)
    dw = 1. / size[0]  # 1./w
    dh = 1. / size[1]  # 1./h
    x = (box[0] + box[1]) / 2.0  # （xmin+xmax）/2.0 物体在图中的中心点x坐标
    y = (box[2] + box[3]) / 2.0  # （ymin+ymax）/2.0物体在图中的中心点y坐标
    w = box[1] - box[0]  # （xmax-xmin） 物体实际像素宽度
    h = box[3] - box[2]  # （ymax-ymin） 物体实际像素高度
    x = x * dw  # 物体中心点x的坐标比（相当于 x/原图w）
    w = w * dw  # 物体宽度的宽度比（相当于 w/原图w）
    y = y * dh  # 物体中心点y的坐标比（相当于 y/原图h）
    h = h * dh  # 物体高度的高度比（相当于 h/原图h）
    return (x, y, w, h)  # 返回相对于原图的物体中心的（x坐标比，y坐标比，宽度比， 高度比），取值范围[0-1]


# Label格式转化
def convert_annotation(image_id):
    # 根据图片名称（image_id）获取对应的xml文件
    in_file = open(r'D:/DYQ/YOLOV520201222/yolov5-predataset/VOCdevkit/VOC2007/Annotations/%s.xml' % (image_id))

    # 生成图片名称（image_id）的txt格式的标签文件（label）的保存路径
    out_file = open(r'D:/DYQ/YOLOV520201222/yolov5-predataset/train/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)  # 解析xml文件
    root = tree.getroot()  # 获取xml文件的根节点
    size = root.find('size')  # 获取指定节点的图像尺寸
    w = int(size.find('width').text)  # 获取图像的宽
    h = int(size.find('height').text)  # 获取图像的高

    for obj in root.iter('object'):  # 标注的物体
        cls = obj.find('name').text  # xml里的name参数（类别名称）
        if cls not in classes:       # 判断从xml获取的 类别标签  是否在上述定义的classes中
            continue
        #cls_id = classes.index(cls)  # 获取索引
        xmlbox = obj.find('bndbox')  # 获取标注物体（xmin，ymin，xmax，ymax）
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
        # b = (xmin, xmax, ymin, ymax)
        #bb = convert((w, h), b)
        bb = b
        out_file.write(str(cls) + " " + " ".join([str(a) for a in bb]) + '\n')


# 获取路径下 图片id
image_ids_train = open(
    r'D:/DYQ/YOLOV520201222/yolov5-predataset/VOCdevkit/VOC2007/ImageSets/Main/train.txt').read().strip().split()
# strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
# split() 通过指定分隔符对字符串进行切片 (默认为所有的空字符，包括空格、换行(\n)、制表符(\t)等)
# image_ids_train = ['2020_08_12_17_45_IMG_5050', '2020_08_12_17_45_IMG_5051', .....]

for image_id in image_ids_train:
    convert_annotation(image_id)  # 转化标注文件格式
