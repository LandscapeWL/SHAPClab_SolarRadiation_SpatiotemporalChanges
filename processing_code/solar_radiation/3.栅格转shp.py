from osgeo import gdal, ogr, osr
import os,time
import geopandas as gpd

def process(png_path,png):
    inraster = gdal.Open(png_path)  #读取路径中的栅格数据
    inband = inraster.GetRasterBand(1)  #这个波段就是最后想要转为矢量的波段，如果是单波段数据的话那就都是1
    prj = osr.SpatialReference()
    prj.ImportFromWkt(inraster.GetProjection())   #读取栅格数据的投影信息，用来为后面生成的矢量做准备

    outshp = "{}/{}.shp".format(shp_path,png[:-4])#给后面生成的矢量准备一个输出文件名，这里就是把原栅格的文件名后缀名改成shp了
    drv = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(outshp):  #若文件已经存在，则删除它继续重新做一遍
        drv.DeleteDataSource(outshp)
    Polygon = drv.CreateDataSource(outshp)  #创建一个目标文件
    Poly_layer = Polygon.CreateLayer(outshp[:-4], srs = prj, geom_type = ogr.wkbMultiPolygon) #对shp文件创建一个图层，定义为多个面类
    newField = ogr.FieldDefn('value',ogr.OFTReal)  #给目标shp文件添加一个字段，用来存储原始栅格的pixel value
    Poly_layer.CreateField(newField)

    gdal.FPolygonize(inband, None, Poly_layer, 0) #核心函数，执行的就是栅格转矢量操作
    Polygon.SyncToDisk()
    del Polygon
    gdf = gpd.read_file(outshp)
    gdf['geometry'] = gdf['geometry'].scale(yfact=-1,origin=(163,163))
    gdf.to_file(outshp)

if __name__ == "__main__":
    t1 = time.time()
    fisheye_path = './data/5.streetview_fisheye_split/2013'
    shp_path = './data/6.jpg_to_shp/2013'

    # 获取已经处理过的图像编号
    exist_list = []
    for _, _, exist_images in os.walk(shp_path):
        for exist_image in exist_images:
            exist_list.append(exist_image.split('.')[0])

    for i,v,m in os.walk(fisheye_path):
        for png in m:
            if png.split('.')[0] in exist_list:                          # 判断数据是否已经处理过, 处理过则跳过
                print('该图片 %s 已经处理过' % (png))
                continue
            png_path = "{}/{}".format(fisheye_path, png)
            process(png_path,png)
            print('{} 完成'.format(png))
    t2 = time.time()
    print('耗时{:.2f}秒'.format(t2-t1))