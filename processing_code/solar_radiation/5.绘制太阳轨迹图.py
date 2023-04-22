from pysolar.solar import *
import datetime
import time
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_angle_dic(_images_path):
    image_list = []
    # 读取 csv 文件
    with open(_images_path, mode='r', encoding='utf-8') as f:
        lines = csv.reader(f)
        for num,line in enumerate(lines):
            if num == 0:  # 跳过表头
                continue
            image_list.append(line)
        return image_list

class UTC(datetime.tzinfo):
    """UTC时区设置"""
    def __init__(self,offset = 0):
        self._offset = offset
    def utcoffset(self, dt):
        return datetime.timedelta(hours=self._offset)
    def tzname(self, dt):
        return "UTC +%s" % self._offset
    def dst(self, dt):
        return datetime.timedelta(hours=self._offset)


def solar_trajectory_mapping(_lng,_lat,_images_name,date_,_clor):
    # 计算24小时的太阳高度及位置 使用列表储存
    altitude_and_azimuth = []
    for h in range(24):
        for m in range(0,60,10):
            date_time = date_ + (h,m)
            date = datetime.datetime(*date_time,tzinfo=UTC(8))
            lng, lat = float(_lng), float(_lat)

            azimuth = get_azimuth(lat, lng, date)   # 计算太阳方位角
            altitude = get_altitude(lat, lng, date) # 计算太阳高度角
            altitude_and_azimuth.append([h,m,altitude,azimuth])
    # 将列表转化为pd格式
    df = pd.DataFrame(altitude_and_azimuth, columns=['hour', 'minute', 'altitude', 'azimuth'])
    # 筛选太阳高度大于0 白天的时间
    dfx = df[df['altitude'] > 0]
    # 将太阳方位和高度 角度转弧度
    dfx['altitude_r'] = (dfx['altitude'] / 360) * 2 * np.pi
    dfx['azimuth_r'] = (dfx['azimuth'] / 360) * 2 * np.pi
    # 重设索引值
    dfx = dfx.reset_index()
    # 获取太阳方位和高度 作为xy坐标
    altitude = dfx['altitude'].values
    azimuth_r = dfx['azimuth_r'].values

    # 绘制图片
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(12, 12),dpi=300)
    # 设置刻度字体大小, 设置刻度和坐标轴距离
    ax.tick_params(labelsize=20,pad=20)
    # 设置y轴的范围
    ax.set_ylim(0, 90)
    # 反转y轴的数值
    ax.invert_yaxis()
    # 设置y轴的刻度
    ax.set_yticks([0, 30, 60])
    # 设置顺时针为正方形
    ax.set_theta_direction(-1)
    # 将0刻度设置为朝向N
    ax.set_theta_zero_location('N')
    # 第一个参数是长度 第二个是参数是角度
    ax.scatter(azimuth_r, altitude, marker='o', color=_clor)

    # ax.text(文本的x轴坐标, 文本的y轴坐标, 要添加的文本
    # color:文本颜色 # size:文本大小 # ha:水平对齐'center','right','left'
    # va:垂直对齐'center','top','bottom', 'baseline' # rotation:文本旋转角度,可以是代表角度的整数,也可以是字符串：'vertical', 'horizontal'
    # bbox:是否将文本放置于文本框内,需要提供一个定义文本框样式的字典
    # style:字体样式,'normal', 'italic', 'oblique'
    for i, v in enumerate(['N', 'E', 'S', 'W']):
        ax.text(i * 0.5 * np.pi, -18, v, va='center', ha='center', fontsize=30, fontfamily='Times New Roman')
    # 保存图片
    plt.savefig('./data/7.solar_trajectory/{}.jpg'.format(_images_name))


if __name__ == "__main__":
    t1 = time.time()
    images_path = './data/0.streetview_info.csv'   # 经纬度储存的csv文件
    # 设定计算太阳的轨迹的日期
    year = 2019
    month = 10
    day = 15
    # 设定绘制点的颜色
    # 'red'红  ‘darkseagreen'灰绿  ‘orange’橙  'purple'紫  'turquoise'蓝  'lightpink'粉
    clor = 'lightpink'

    date = (year, month, day)
    images_list = get_angle_dic(images_path)       # 获取所有的旋转角度储存的列表
    for i in range(len(images_list)):
        images_name = images_list[i][0]
        lng = images_list[i][1]
        lat = images_list[i][2]
        print('开始处理:',images_name)
        # 通过经纬度和文件名绘制太阳轨迹
        solar_trajectory_mapping(lng,lat,images_name,date,clor)




