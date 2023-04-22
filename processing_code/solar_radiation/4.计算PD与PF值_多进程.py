from pysolar.solar import *
import datetime
import pandas as pd
import numpy as np
import geopandas as gpd
import math
import time
import os
from pysolar import radiation
from shapely.geometry import Point,LineString
from shapely.ops import polygonize
import warnings
warnings.filterwarnings('ignore')
import csv
from multiprocessing import Process, Queue



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


def process_shp(streetview_shp):
    """
    处理经过语义分割之后的shp文件，包括：
    1. 原点位置为（0,0）；
    2. 按极坐标图进行缩放。
    """
    gdf = gpd.read_file(streetview_shp)

    gdf['geometry'] = gdf['geometry'].translate(xoff=-326 / 2, yoff=-326 / 2)
    gdf['geometry'] = gdf.scale(xfact=180/326, yfact=180/326, origin=(0, 0))
    streetview_gdf = gdf[gdf['value'] != 0]
    return streetview_gdf


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


def get_PD_Radiation(lng,lat,streetview_gdf,date_):
    """计算直射辐射比例"""
    altitude_and_azimuth = []
    for h in range(24):
        for m in range(0,60,10):
            date_time = date_ + (h,m)
            date = datetime.datetime(*date_time,tzinfo=UTC(8))
            lng, lat = float(lng), float(lat)

            azimuth = get_azimuth(lat, lng, date)   # 计算太阳方位角
            altitude = get_altitude(lat, lng, date) # 计算太阳高度角
            solar_radiation = radiation.get_radiation_direct(date, altitude) # 计算太阳辐射量

            # 列表记录着 [[小时, 分钟, altitude, azimuth],[小时, 分钟, altitude, azimuth],...]
            altitude_and_azimuth.append([h,m,altitude,azimuth,solar_radiation])

    df = pd.DataFrame(altitude_and_azimuth, columns=['hour', 'minute', 'altitude', 'azimuth', 'solar_radiation'])
    # 筛选altitude为正 筛选出白天的
    dfx = df[df['altitude'] > 0]
    # 将角度制转为弧度制
    dfx['altitude_r'] = (dfx['altitude'] / 360) * 2 * np.pi
    dfx['azimuth_r'] = (dfx['azimuth'] / 360) * 2 * np.pi
    dfx = dfx.reset_index()
    # 计算白天24小时中平均的太阳辐射能量
    mean_solar_radiation = dfx['solar_radiation'].mean()


    # 通过方位角和高度角 计算二维平面位置
    def convert(x):
        altitude = x[0]
        azimuth = x[1]
        if azimuth < 90:
            azimuth = 90 - azimuth
            vx = np.cos((azimuth / 360) * 2 * np.pi) * (90 - altitude)
            vy = np.sin((azimuth / 360) * 2 * np.pi) * (90 - altitude)
        elif azimuth >= 90 and azimuth < 180:
            azimuth = azimuth - 90
            vx = np.cos((azimuth / 360) * 2 * np.pi) * (90 - altitude)
            vy = -np.sin((azimuth / 360) * 2 * np.pi) * (90 - altitude)
        elif azimuth >= 180 and azimuth < 270:
            azimuth = azimuth - 180
            vx = -np.sin((azimuth / 360) * 2 * np.pi) * (90 - altitude)
            vy = -np.cos((azimuth / 360) * 2 * np.pi) * (90 - altitude)
        else:
            azimuth = azimuth - 270
            vx = -np.cos((azimuth / 360) * 2 * np.pi) * (90 - altitude)
            vy = np.sin((azimuth / 360) * 2 * np.pi) * (90 - altitude)
        return vx, vy
    dfx['xy'] = dfx[['altitude','azimuth']].apply(convert,axis=1)

    dfp = gpd.GeoDataFrame({'geometry':[Point(i) for i in dfx['xy'].values],
                           'hour':[dfx['hour'][i] for i in range(len(dfx))],
                            'minute':[dfx['minute'][i] for i in range(len(dfx))],
                            'altitude':[dfx['altitude'][i] for i in range(len(dfx))],
                            'azimuth':[dfx['azimuth'][i] for i in range(len(dfx))],
                            'altitude_r':[dfx['altitude_r'][i] for i in range(len(dfx))],
                            'azimuth_r':[dfx['azimuth_r'][i] for i in range(len(dfx))]
                           })
    # 将太阳位置信息和矢量图进行连接
    df_fl = gpd.sjoin(dfp,streetview_gdf,how='left',op='within')
    df_fl['cos_altitude_r'] = df_fl['altitude_r'].apply(lambda x:np.cos(x))

    PD = np.nansum(df_fl.dropna(subset=['index_right'])['cos_altitude_r']) / np.nansum(df_fl['cos_altitude_r'])
    return PD,mean_solar_radiation

def get_PF(streetview_gdf):
    """计算漫射辐射比例"""
    # 设置天区
    lstx = [i * 90 / 8 for i in range(1, 9)]
    mls = [LineString(Point(0, 0).buffer(i).exterior.coords[:]) for i in lstx] + \
          [LineString(((0, 0), (math.cos((22.5 * i / 360) * 2 * math.pi) * 95,
                                math.sin((22.5 * i / 360) * 2 * math.pi) * 95))) for i in range(16)]

    gmls = gpd.GeoSeries(mls,name='geometry').reset_index()
    polar = list(polygonize(gmls['geometry'].unary_union))
    gdf = gpd.GeoDataFrame(polar,columns=['geometry'])
    gdf['area'] = gdf['geometry'].area

    # 这里判断天空是否为空, 如果是空返回个0
    if streetview_gdf.empty:
        PF = 0
    else:
        dfv = gpd.overlay(streetview_gdf, gdf, how='identity')
        dfv['area2'] = dfv['geometry'].area
        dfv['inner_zenith'] = dfv['geometry'].apply(lambda x: 90 - Point(0, 0).distance(x))
        dfv['outer_zenith'] = dfv['geometry'].apply(lambda x: 90 - Point(0, 0).hausdorff_distance(x))
        dfv['centroid_zenith'] = dfv['geometry'].\
            apply(lambda x: 90 - Point(0, 0).distance(Point(x.centroid.x, x.centroid.y)))

        def get_pf(x):
            gaz = x['area2'] / x['area']
            theta2 = (x['outer_zenith'] / 360) * 2 * np.pi
            theta1 = (x['inner_zenith'] / 360) * 2 * np.pi
            thetaz = (x['centroid_zenith'] / 360) * 2 * np.pi
            cos_theta2_1 = np.cos(theta2) - np.cos(theta1)
            cos_thetaz = np.cos(thetaz)
            return gaz * cos_theta2_1 * cos_thetaz

        dfv['vl'] = dfv.apply(get_pf,axis=1)
        PF = np.nansum(dfv['vl']) / 16
    return PF

def set_csv(_months, _result_name):
    # 设置表头
    headers = ['name', 'PF', ]
    for id in range(len(_months)):
        head = 'PD_%d' % (_months[id])
        headers.append(head)
    _exist_list = []
    # 创建csv文件及表头
    if not os.path.exists(_result_name):
        with open('%s' % _result_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    else:
        # 读取csv中已经存在的数据
        with open(_result_name, 'r', newline='') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                _exist_list.append(row[0])
    # print(exist_list)
    return _exist_list

# 设置要采集街景的多线程序列
def set_sequence(_q,_images_list):
    for i in range(len(_images_list)):
        # 获取id,经纬度坐标
        FID, get_wgs_x, get_wgs_y = _images_list[i][0], _images_list[i][1], _images_list[i][2]
        # 设置序列
        _q.put([FID, get_wgs_x, get_wgs_y])

def start(_q,_exist_list, _shp_path, _images_path, _result_name, _year, _months, _day):
    while not _q.empty():
        positon = _q.get()
        FID, lng, lat = positon[0], positon[1], positon[2]
        # 判断数据是否已经处理过, 处理过则跳过
        if FID in _exist_list:
            print('该序号 %s 已经处理过' % (FID))
            continue
        print('开始处理:',FID)

        shp_file = "{}/{}.shp".format(_shp_path, FID)
        # 筛选出矢量文件的天空区域
        streetview_gdf = process_shp(shp_file)
        # 计算pd和pf值
        vl_PF = get_PF(streetview_gdf)
        result_list = [FID,vl_PF]
        # 每个月计算一次PD
        for month in _months:
            date = (_year, month, _day)
            vl_PD, solar_radiation = get_PD_Radiation(lng, lat, streetview_gdf,date)
            # 构建保存的结果列表
            result_list.append(vl_PD)

        print(result_list)
        # 将结果写入csv
        with open('%s' % _result_name, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(result_list)

        print(' 处理结果:',result_list)
        print(' 处理完毕:',FID)
        print('---'*10)

if __name__ == "__main__":

    # 设定计算太阳的轨迹的日期
    year = 2013
    # 需要处理哪几个月
    months = [5,6,7,8,9,10]
    day = 15
    # 存储的栅格转shp后的文件夹
    shp_path = './data/6.jpg_to_shp/2013'
    # 鱼眼图旋转角度储存的文件位置
    images_path = './data/0.historical_svid_2013.csv'
    # 结果文件名字
    result_name = './data/8.pd&pf_{}.csv'.format(year)
    # 进程数
    process_num = 1

    # 获取所有的旋转角度储存的列表
    images_list = get_angle_dic(images_path)
    # 创建和设置csv文件, 读取已经存在的
    exist_list = set_csv(months, result_name)

    # 设置多线程序列及进程锁
    q = Queue()
    # 设置序列
    set_sequence(q,images_list)
    # 创建多线程阻塞记录列表
    process_list = []
    for i in range(0, process_num):
        # 创建线程
        process = Process(target=start, args=(q,exist_list, shp_path, images_path, result_name, year, months, day))
        print("-->进程", i)
        process_list.append(process)
        process.start()
    for t in process_list:
        t.join()
    while not q.empty():
        pass

    print(' [-] 本次太阳辐射处理完成 谢谢使用\n\n                   __Development by Phd.WangLei')


