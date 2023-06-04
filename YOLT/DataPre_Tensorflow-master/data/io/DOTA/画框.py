import random
import cv2

       
if __name__ == '__main__':
    xyxy = [491+512, 580+512, 509+512, 616+512]
    # file_name = 'D:/YOLT/20210128175159_512_512_640_640_0_2309_1732.jpg'  # 文件路径
    file_name = 'D:/YOLT/20210124104456.jpg'
    img = cv2.imread(file_name)  # 读取文件
    
    names = ['car', 'people', 'tie']
    conf = 0.93
    cls = 0
    label = '%s %.2f' % (names[int(cls)], conf)
    
    colors = [187, 83, 195]

    x = xyxy
    label = label
    color = colors
    line_thickness = 3

    # tl=3
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

    color = color or [random.randint(0, 255) for _ in range(3)]
    
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

    cv2.imwrite('eee.jpg',img)
    cv2.resizeWindow(file_name,800,800)
    cv2.imshow(file_name, img)
    cv2.waitKey(20000)
    '''
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        cv2.imshow(img)
    '''
