import cv2
import numpy as np
from math import pi, atan
import os,time
import csv


def pano_process(_split_file,_north_angle):
    fisheye_dst = "{}/{}".format(fisheye_path, image)

    _img = cv2.imread(_split_file, 1)
    height, width = _img.shape[:2]
    cx = width / (2*pi)
    cy = width / (2*pi)
    img_hemi = np.zeros((int(cx + 1) * 2, int(cx + 1) * 2, 3), dtype=np.uint8)

    for col in range(img_hemi.shape[1]):
        for row in range(img_hemi.shape[0]):
            if row < cx:
                theta = pi / 2 + atan((col - cy) / (row - cx))
            else:
                theta = (pi * 3) / 2 + atan((col - cy) / (row - cx))

            r = np.sqrt((row - cx) ** 2 + (col - cy) ** 2)

            x = (theta * width) / (2 * pi)
            y = (r * height) / cx
            img_hemi[row][col] = cv2.getRectSubPix(_img, (1, 1), (x, y))

    # 设置黑色的掩膜
    mask = np.zeros_like(img_hemi)
    wd = int(cx)
    mask = cv2.circle(mask, (wd, wd), wd, (252, 255, 255), -1)
    result = cv2.bitwise_and(img_hemi, mask)

    # 翻转鱼眼图方向
    result = cv2.flip(result, 1, dst=None)
    # 图像旋转变换矩阵 参数为：旋转中心、旋转角度、缩放比例
    rot_mat = cv2.getRotationMatrix2D((int(cx + 1), int(cx + 1)), _north_angle, 1)
    # 仿射变换 旋转朝向北
    rotated = cv2.warpAffine(result, rot_mat, (result.shape[1], result.shape[0]))

    # 以下两行二选一运行
    # cv2.imwrite(fisheye_dst, img_hemi)   # 写入未处理图像
    cv2.imwrite(fisheye_dst, rotated)   # 写出图像
    print('{} done!'.format(fisheye_dst))

def get_angle_dic(_angle_path):
    angle_dic = {}
    # 读取 csv 文件
    with open(_angle_path, mode='r', encoding='utf-8') as f:
        lines = csv.reader(f)
        for num,line in enumerate(lines):
            if num == 0:  # 跳过表头
                continue
            angle_dic[line[0]] = line[6]
        return angle_dic

def get_north_angle(_angle_dic, _image):
    _image = _image.split('.')[0]       # 获得图片的序号
    _north_angle = _angle_dic['{}'.format(_image)]  # 通过序号获得旋转角度
    _north_angle = float(_north_angle)  # 将旋转角度从str 转为float
    if _north_angle <= 180:
        _north_angle = 180 - _north_angle
    else:
        _north_angle = 540 - _north_angle

    return _north_angle


if __name__ == '__main__':
    t1 = time.time()
    # 以下需要运行两次 生成原图鱼眼和分割后的鱼眼
    # 生成原图的鱼眼图
    # split_path = './data/2.streetview_clip/2013'      # 裁剪后的街景（原图）
    # fisheye_path = './data/4.streetview_fisheye_clip/2013'  # 创建鱼眼图后，导出到那个文件夹
    # 生成图分割的鱼眼图
    split_path = './data/3.streetview_split/2013'      # 裁剪后的街景（图分割）
    fisheye_path = './data/5.streetview_fisheye_split/2013'  # 创建鱼眼图后，导出到那个文件夹

    angle_path = './data/0.historical_svid_2013.csv'     # 鱼眼图旋转角度储存的文件位置
    # 获取所有的旋转角度储存的字典
    angle_dic = get_angle_dic(angle_path)

    # 获取已经处理过的图像编号
    exist_list = []
    for _, _, exist_images in os.walk(fisheye_path):
        for exist_image in exist_images:
            exist_list.append(exist_image)


    for i,j,images in os.walk(split_path):                   # 遍历街景文件夹中的跟目录
        for image in images:                                 # 循环每一张街景图片
            if image in exist_list:                          # 判断数据是否已经处理过, 处理过则跳过
                print('该图片 %s 已经处理过' % (image))
                continue
            split_file = "{}/{}".format(split_path, image)   # 组合街景文件的路径
            north_angle = get_north_angle(angle_dic, image) # 读取每个街景图旋转的角度
            # print(north_angle)
            pano_process(split_file,north_angle)             # 调用函数，执行处理
    t2 = time.time()
    print('耗时{:.2f}秒'.format(t2-t1))