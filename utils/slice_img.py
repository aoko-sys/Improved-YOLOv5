import cv2
import time
from pathlib import Path
def slice_img(image_path, outdir, sliceWidth, sliceHeight):
        # image_path = 'D:/DYQ/DOTA/YOLT/yolt-master/test_images/header.jpg'
        path = Path(image_path)
        out_name = path.stem
        print("out_name:", out_name)
        outdir = outdir + '/'
        # sliceHeight = 416
        # sliceWidth = 416
        zero_frac_thresh = 0.5
        overlap = 0.6    # 重叠率
        verbose = False
        image0 = cv2.imread(image_path, 1)   # color 读取图片
        ext = '.' + image_path.split('.')[-1]  # .jpg  与原始图片保持一样的格式
        win_h, win_w = image0.shape[:2]   # 原始图片的width  height
        # if slice sizes are large than image, pad the edges 如果切片大小比图像大，填充边缘
        pad = 0
        if sliceHeight > win_h:
                pad = sliceHeight - win_h
        if sliceWidth > win_w:
                pad = max(pad, sliceWidth - win_w)  # 找w 和 h的最大填充值
        # pad the edge of the image with black pixels 用黑色像素填充图像的边缘
        win_size = sliceHeight*sliceWidth  # 416*416
        t0 = time.time()
        n_ims = 0
        n_ims_nonull = 0
        # overlap=0.2 滑动窗口的重叠率
        dx = int((1. - overlap) * sliceWidth)   # 332
        dy = int((1. - overlap) * sliceHeight)  # 332
        print('image0',image0.shape)
        for y0 in range(0, image0.shape[0], dy):
                for x0 in range(0, image0.shape[1], dx):  # sliceWidth):
                        n_ims += 1
                        # make sure we don't have a tiny image on the edge  确保边缘没有小图像
                        if y0 + sliceHeight > image0.shape[0]:
                                y = image0.shape[0] - sliceHeight
                        else:
                                y = y0
                        if x0 + sliceWidth > image0.shape[1]:
                                x = image0.shape[1] - sliceWidth
                        else:
                                x = x0
                        print('y0:%d,x0:%d' % (y0, x0))
                        # extract image   # 提取图像
                        window_c = image0[y:y + sliceHeight, x:x + sliceWidth]
                        window = cv2.cvtColor(window_c, cv2.COLOR_BGR2GRAY)
                        ret, thresh1 = cv2.threshold(window, 2, 255, cv2.THRESH_BINARY)
                        non_zero_counts = cv2.countNonZero(thresh1)  # 对二值化图像，可得到非零像素点
                        zero_counts = win_size - non_zero_counts
                        zero_frac = float(zero_counts) / win_size
                        if zero_frac >= zero_frac_thresh:
                                if verbose:
                                        print("Zero frac too high at:", zero_frac)
                                continue
                                # else save
                        else:
                                imgname = out_name + '_' + str(y) + '_' + str(x) + '_' + str(sliceHeight) + '_' + str(
                                        sliceWidth) + '_' + str(pad) + '_' + str(win_w) + '_' + str(win_h) + ext
                                print("imaname:", imgname)
                                cv2.imwrite(outdir+imgname, window_c)

                                # 展示切图效果
                                #cv2.imshow('image', window_c)
                                #cv2.waitKey(20)
                                n_ims_nonull += 1
                print("Num slices:", n_ims, "Num non-null slices:", n_ims_nonull,
                      "sliceHeight", sliceHeight, "sliceWidth", sliceWidth)
                print("Time to slice", image_path, time.time() - t0, "seconds")