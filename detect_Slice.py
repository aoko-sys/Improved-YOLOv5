import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.slice_img import slice_img
from utils.show_chip import showChip
from utils.datasets_chips import LoadImagesChip


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://'))
    # Directories 目录 创建
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    # Initialize
    set_logging()
    device = select_device(opt.device)
    # device = device(type='cuda', index=0)
    half = device.type != 'cpu'  # half precision only supported on CUDA 仅在CUDA上支持半精度
    print(1)
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    print(2)
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size model.stride = tensor([ 8., 16., 32.] 下采样倍数
    # 判断预定图片尺寸  是否  是32的倍数
    print(3)
    if half:
        model.half()  # to FP16
    print(4)
    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference 设置为True可以加速常量图像大小的推断
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        # print('source:',source)   # source: data/images
        dataset = LoadImages(source, img_size=imgsz)
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names  # bbox 类别名字
    print("names:",names)
    names=['cotter','nest','missinsulator']
    # hasattr(model, 'module')函数用于判断对象是否包含对应的属性
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]  # bbox 颜色
    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        # path: D:\DYQ\YOLOV520201222\yolov5-master\data\images\20210128175159.jpg
        # img: (3, 480, 640)
        # im0s: (1732, 2309, 3)
        # ----------yolt（1） start--------
        print("图片路径:",path)
        import os
        outdir = os.path.abspath(opt.outdir)  # absolute path 绝对路径
        # outdir: D:\DYQ\YOLOV520201222\yolov5-master\data\SliceImage
        # 首先创建文件夹 存储chip图片
        import shutil
        if os.path.exists(outdir):
            shutil.rmtree(outdir)    # 删除文件

        if not os.path.exists(outdir):
            os.makedirs(outdir)      # 创建文件
        for ss in [960]:
            print("切图：")
            slice_img(path, outdir, ss, ss)

        datasetChip = LoadImagesChip(outdir, img_size=imgsz)  # 加载高分辨率的chip图片
        # 将所有chip的bbox放到一起
        detchip_all = []
        for pathchip, imgchip, img0schip in datasetChip:
            # pathchip: D:\DYQ\YOLOV520201222\yolov5-master\data\SliceImage\20210128175159_0_0_640_640_0_2309_1732.jpg
            # imgchip: (3, 640, 640)
            # img0schip: (640, 640, 3)
            chipname = Path(pathchip).stem  # ['20210128175159', '0', '0', '640', '640', '0', '2309', '1732']
            namexywhchip = chipname.split('_')
            y = int(namexywhchip[1])  # 当前chip坐标相对高分率图片的坐标偏移量
            x = int(namexywhchip[2])
            imgchip = torch.from_numpy(imgchip).to(device)
            imgchip = imgchip.half() if half else img.float()
            imgchip /= 255.0
            if imgchip.ndimension() == 3:
                imgchip = imgchip.unsqueeze(0)
                # imgchip: torch.Size([1, 3, 640, 640])
            t1 = time_synchronized()
            predchip = model(imgchip, augment=opt.augment)[0]  # # torch.Size([1, 18900, 85])
            print("名字：", Path(pathchip).name)


            # 将坐标还原
            for ichip, detchip in enumerate(predchip):  # detections per image
                # Rescale boxes from img_size to im0 size 将方框大小从img_size rescale为im0大小
                detchip[:, :4] = scale_coords(imgchip.shape[2:],  # 预处理后[384, 640] 图像尺寸
                                              detchip[:, :4],
                                              img0schip.shape  # (720, 1280, 3) 原始图片大小
                                              ).round()  # round()函数  四舍五入
                # detchip: torch.Size([25200, 85])
                detchip[:, 0] = detchip[:, 0] + x
                detchip[:, 1] = detchip[:, 1] + y
                # 640图片只检测cotter
                print('#####')
                print(detchip)
                detchip_all.append(detchip)
        if len(detchip_all):
            # Apply NMS
            print("before:", detchip_all)
            detchip_all = torch.stack(detchip_all)  # stack形状: torch.Size([20, 25200, 85])
            print("stack形状:", detchip_all.shape)
            print("stack:", detchip_all)
            detchip_all = detchip_all.view([1, detchip_all.size(0)*detchip_all.size(1), detchip_all.size(2)])
            print("NSM before形状:",detchip_all.shape)  # torch.Size([1, 504000, 85])
            predchip_all = non_max_suppression(detchip_all,
                                               opt.conf_thres,  # conference阈值
                                               opt.iou_thres,  # iou阈值
                                               classes=opt.classes,  # 设置只保留某一部分类别，形如0或者0 2 3
                                               agnostic=opt.agnostic_nms)  # 进行nms是否也去除不同类别之间的框
            print("NMS:",predchip_all)
            print('----------end-----------------')
            t2 = time_synchronized()
            # Process detections
            for i, det in enumerate(predchip_all):  # detections per image
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                # p: D:\DYQ\YOLOV520201222\yolov5-master\data\images\20210128175159.jpg
                # im0: (1732, 2309, 3)
                # frame: 0
                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                # txt_path = runs\detect\exp45\labels\zidane
                s += '%gx%g ' % im0.shape[:2]  # print string  # 640x480
                # img.shape = torch.Size([1, 3, 640, 480])
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                # im0.shape = (720, 1280, 3)
                # gn = tensor([1280,  720, 1280,  720])
                print('gn:', gn)  # [2309, 1732, 2309, 1732]
                if len(det):
                    # Rescale boxes from img_size to im0 size 将方框大小从img_size rescale为im0大小
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f'{n} {names[int(c)]}s, '  # add to string
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        if save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.3f}s)')
                # Stream results
                #cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                if view_img:
                    cv2.imshow(str(p), im0)
                    if cv2.waitKey(10) == ord('q'):  # q to quit
                        raise StopIteration
                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            fourcc = 'mp4v'  # output video codec
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                        vid_writer.write(im0)
        # ----------yolt（2）--------
        # 下一次检测之前，将前一次chip文件夹进行删除
        # ----------yolt（2）--------
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    print(f'Done. ({time.time() - t0:.3f}s)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/exp50/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.8, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.2, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0,1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--outdir', default='data/SliceImage', help='save chip with image')  # 结果图片chip保存路径
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
