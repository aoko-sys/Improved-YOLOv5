'''
xmin,ymin,xmax,ymax 剪切后的chip是根据中心来定位的，可能会导致，
坐标更改后，新生成的xml文件里面的（xmin,ymin,xmax,ymax ）超过图片边界

这种超出边界，可能会影响结果，具体会不会，我在v1版本，对（xmin,ymin,xmax,ymax ）
去做一个限制，使得（xmin,ymin,xmax,ymax）不会超图像边界
'''
import os
from xml.dom.minidom import Document
import numpy as np
import copy
import cv2
import sys
sys.path.append('../../..')
from help_utils.tools import mkdir


def save_to_xml(save_path, imgname, im_height, im_width, objects_axis, label_name):
    im_depth = 3
    object_num = len(objects_axis)
    doc = Document()

    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    folder = doc.createElement('folder')
    folder_name = doc.createTextNode('VOC2007')
    folder.appendChild(folder_name)
    annotation.appendChild(folder)

    filename = doc.createElement('filename')
    filename_name = doc.createTextNode(imgname)
    filename.appendChild(filename_name)
    annotation.appendChild(filename)

    source = doc.createElement('source')
    annotation.appendChild(source)

    database = doc.createElement('database')
    database.appendChild(doc.createTextNode('The VOC2007 Database'))
    source.appendChild(database)

    annotation_s = doc.createElement('annotation')
    annotation_s.appendChild(doc.createTextNode('PASCAL VOC2007'))
    source.appendChild(annotation_s)

    image = doc.createElement('image')
    image.appendChild(doc.createTextNode('flickr'))
    source.appendChild(image)

    flickrid = doc.createElement('flickrid')
    flickrid.appendChild(doc.createTextNode('322409915'))
    source.appendChild(flickrid)

    owner = doc.createElement('owner')
    annotation.appendChild(owner)

    flickrid_o = doc.createElement('flickrid')
    flickrid_o.appendChild(doc.createTextNode('knautia'))
    owner.appendChild(flickrid_o)

    name_o = doc.createElement('name')
    name_o.appendChild(doc.createTextNode('duan'))
    owner.appendChild(name_o)

    size = doc.createElement('size')
    annotation.appendChild(size)
    width = doc.createElement('width')
    width.appendChild(doc.createTextNode(str(im_width)))
    height = doc.createElement('height')
    height.appendChild(doc.createTextNode(str(im_height)))
    depth = doc.createElement('depth')
    depth.appendChild(doc.createTextNode(str(im_depth)))
    size.appendChild(width)
    size.appendChild(height)
    size.appendChild(depth)
    segmented = doc.createElement('segmented')
    segmented.appendChild(doc.createTextNode('0'))
    annotation.appendChild(segmented)
    for i in range(object_num):
        objects = doc.createElement('object')
        annotation.appendChild(objects)
        object_name = doc.createElement('name')
        object_name.appendChild(doc.createTextNode(label_name[int(objects_axis[i][-1])]))
        objects.appendChild(object_name)
        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode('Unspecified'))
        objects.appendChild(pose)
        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode('1'))
        objects.appendChild(truncated)
        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode('0'))
        objects.appendChild(difficult)
        bndbox = doc.createElement('bndbox')
        objects.appendChild(bndbox)

        x0 = doc.createElement('xmin')
        x0.appendChild(doc.createTextNode(str((objects_axis[i][0]))))
        bndbox.appendChild(x0)
        y0 = doc.createElement('ymin')
        y0.appendChild(doc.createTextNode(str((objects_axis[i][1]))))
        bndbox.appendChild(y0)

        x1 = doc.createElement('xmax')
        x1.appendChild(doc.createTextNode(str((objects_axis[i][2]))))
        bndbox.appendChild(x1)
        y1 = doc.createElement('ymax')
        y1.appendChild(doc.createTextNode(str((objects_axis[i][3]))))
        bndbox.appendChild(y1)

    f = open(save_path, 'w')
    f.write(doc.toprettyxml(indent=''))
    f.close()


def format_label(txt_list):
    format_data = []
    for i in txt_list:
        if len(i.split(' ')) < 4:   # len(i.split(' ')) = 5 (label,x,y,w,h)
            continue
        format_data.append(
            [float(xy) for xy in i.split(' ')[1:]] + [class_list.index(i.split(' ')[0])]
        )   # 将['car 961 1545 1322 1757\n'...]----->[ 961., 1545., 1322., 1757.,    0.]

        if i.split(' ')[0] not in class_list:
            print('warning found a new label :', i.split(' ')[0])
            exit()
    return np.array(format_data)


def clip_image(file_idx, image, boxes_all, width, height, stride_w, stride_h):
    min_pixel = 5
    print(file_idx)

    boxes_all_5 = boxes_all
    print(boxes_all[np.logical_or((boxes_all_5[:, 2]-boxes_all_5[:, 0]) <= min_pixel,
                                  (boxes_all_5[:, 3]-boxes_all_5[:, 1]) <= min_pixel), :])
    # 判断标准物体是否存在像素低于min_pixel的情况

    boxes_all = boxes_all[np.logical_and((boxes_all_5[:, 2]-boxes_all_5[:, 0]) > min_pixel,
                                         (boxes_all_5[:, 3]-boxes_all_5[:, 1]) > min_pixel), :]

    if boxes_all.shape[0] > 0:  # 判断有没有标注物体
        shape = image.shape     # (2160, 3840, 3)
        for start_h in range(0, shape[0], stride_h):      # 原始图片h
            for start_w in range(0, shape[1], stride_w):  # 原始图片w
                boxes = copy.deepcopy(boxes_all)
                box = np.zeros_like(boxes_all)
                start_h_new = start_h
                start_w_new = start_w
                if start_h + height > shape[0]:
                    start_h_new = shape[0] - height
                if start_w + width > shape[1]:
                    start_w_new = shape[1] - width

                top_left_row = max(start_h_new, 0)
                top_left_col = max(start_w_new, 0)
                bottom_right_row = min(start_h + height, shape[0])
                bottom_right_col = min(start_w + width, shape[1])

                subImage = image[top_left_row:bottom_right_row, top_left_col: bottom_right_col]   # 切割

                #
                box[:, 0] = boxes[:, 0] - top_left_col
                box[:, 2] = boxes[:, 2] - top_left_col
                #box[:, 4] = boxes[:, 4] - top_left_col
                #box[:, 6] = boxes[:, 6] - top_left_col

                box[:, 1] = boxes[:, 1] - top_left_row
                box[:, 3] = boxes[:, 3] - top_left_row
                #box[:, 5] = boxes[:, 5] - top_left_row
                #box[:, 7] = boxes[:, 7] - top_left_row
                #box[:, 8] = boxes[:, 8]
                box[:, 4] = boxes[:, 4]

                center_y = 0.5 * (box[:, 1] + box[:, 3])  # center_y: [1651.  1045.5]
                center_x = 0.5 * (box[:, 0] + box[:, 2])  # center_x: [1141.5 2643. ]

                cond1 = np.intersect1d(np.where(center_y[:] >= 0)[0], np.where(center_x[:] >= 0)[0])  # 一定成立
                cond2 = np.intersect1d(np.where(center_y[:] <= (bottom_right_row - top_left_row))[0],  # (ymax-ymin)
                                       np.where(center_x[:] <= (bottom_right_col - top_left_col))[0])  # (xmax-xmin)
                idx = np.intersect1d(cond1, cond2)

                if len(idx) > 0 and (subImage.shape[0] > 5 and subImage.shape[1] > 5):
                    mkdir(os.path.join(save_dir, 'images'))
                    img = os.path.join(save_dir, 'images', "%s_%04d_%04d.jpg" % (file_idx, top_left_row, top_left_col))
                    print('img:', img)

                    cv2.imwrite(img, subImage)    # 保存切割的图片
                    cv2.imshow("image", subImage)
                    cv2.waitKey(20000)
                    mkdir(os.path.join(save_dir, 'labeltxt'))
                    xml = os.path.join(save_dir, 'labeltxt',
                                       "%s_%04d_%04d.xml" % (file_idx, top_left_row, top_left_col))
                    print("xml:", xml)
                    imgname = "%s_%04d_%04d.jpg" % (file_idx, top_left_row, top_left_col)
                    print('imgname:', imgname)
                    save_to_xml(xml, imgname, subImage.shape[0], subImage.shape[1], box[idx, :], class_list)


if __name__ == '__main__':
    class_list = ['car']                  # 要检测的类别标签
    print('class_list', len(class_list))

    raw_data = 'D:/YOLT'
    raw_images_dir = os.path.join(raw_data, 'images')   # 图片
    raw_label_dir = os.path.join(raw_data, 'labelTxt')  # xml转成txt文件 txt（cls，xmin，ymin，xmax，ymax）

    save_dir = 'D:/YOLT/trianval/'    # 将image剪裁成chip后保存路径

    images = [i for i in os.listdir(raw_images_dir) if 'jpg' in i]  # 遍历文件夹，获取子文件（图片）名称
    labels = [i for i in os.listdir(raw_label_dir) if 'txt' in i]

    print('find image', len(images))
    print('find label', len(labels))

    img_h, img_w = 640, 640

    overlap = 0.3  # chip重叠率
    stride_w = int((1. - overlap) * img_w)   # 448
    stride_h = int((1. - overlap) * img_h)   # 448

    for idx, img in enumerate(images):
        print(idx, 'read image', img)
        img_data = cv2.imread(os.path.join(raw_images_dir, img))  # cv2读取图片

        txt_data = open(os.path.join(raw_label_dir, img.replace('jpg', 'txt')), 'r').readlines()  # 读取与图片相应的txt文件
        box = format_label(txt_data)
        # box = array([[ 961., 1545., 1322., 1757.,    0.],    (xmin, ymin, xmax, ymax, cls)
        #              [2598.,  989., 2688., 1102.,    0.]])

        if box.shape[0] > 0:
            clip_image(img.strip('.jpg'),  # 图片id
                       img_data,  # 图片
                       box,       # gt
                       img_w,     # chip_w
                       img_h,     # chip_h
                       stride_w,
                       stride_h)
