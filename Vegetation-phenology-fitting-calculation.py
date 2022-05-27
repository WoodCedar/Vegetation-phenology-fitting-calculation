from osgeo import gdal
import sys
import os
import numpy as np
from scipy.optimize import curve_fit

def smooth(y, box_pts): #sg滤波函数
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def f_gauss(x, A, B, sigma): #高斯函数
    return A*np.exp(-(x-B)**2/(2*sigma**2)) 



def index_number(li,defaultnumber): #返回最接近的一个查找值
    number = float(defaultnumber)
    select1 = 999999.0
    index1 = 0
    for i in range(len(li)):
        select =  abs(li[i]-number)
        if (abs(select1) - abs(select)<0.001):
                select1 = select
                index1 = i
    return index1 #,li[index1]

def max_number(li):#返回最大值
    max = -9999.0
    max_index = 0
    for i in range(len(li)):
        if(li[i]>max):
            max = li[i]
            max_index=i
    return  max_index#max,

def max_rate_number(li):#返回斜率最大值
    max = abs(li[1]-li[0])
    max_index = 1
    for i in range(2,len(li)):
        if(abs(li[i]-li[i-1])>max):
            max = abs(li[i]-li[i-1])
            max_index=i
    return  max_index#max,

def day_number(li,daynumber): #返回x天的数值
    return  li[daynumber]

def Two_index_number(li,defaultnumber): #返回正态函数左右两侧的最接近的数值
    max = -9999
    max_index = 0
    for i in range(len(li)):
        if (li[i] > max):
            max = li[i]
            max_index = i
    if max == defaultnumber:
        print("您要查找的就是最大值")
        return max_index
    number = float(defaultnumber)
    select1 = 999999
    index1 = 0
    for i in range(0,max_index):
        select = abs(li[i] - number)
        if (abs(select1) > abs(select)):
            select1 = select
            index1 = i
    result_1 = index1
    number = float(defaultnumber)
    select1 = 999999.0
    index1 = 0
    for i in range(max_index,len(li)):
        select = abs(li[i] - number)
        if (abs(select1) > abs(select)):
            select1 = select
            index1 = i
    result_2 = index1
    return result_1,result_2


def SOS(li):#左侧
    max = -9999
    max_index =-1
    for i in range(len(li)):
        if (li[i] > max):
            max = li[i]
            max_index = i#找到最大值
    select1 = 0;index1 = 0
    for i in range(1, max_index):
        tmp = li[i]-li[i-1]
        if (tmp > select1):
            select1 =tmp
            index1 = i
    result = index1+1
    return result
def EOS(li):
    max = -9999
    max_index = -1
    for i in range(len(li)):
        if (li[i] > max):
            max = li[i]
            max_index = i
    select1 = li[max_index]-li[max_index-1]
    index1 = li[max_index]
    for i in range(max_index, len(li)):
        tmp = li[i] - li[i - 1]
        if (tmp < select1):
            select1 = tmp
            index1 = i
    result = index1 + 1
    return result

def read_filename(file_list,tif_listname): #读取有多少个tif的函数,并获取到列数和行数
    for file in file_list:
        if os.path.splitext(file)[1] == ".tif":
            name = file_path + '/' + os.path.splitext(file)[0] + os.path.splitext(file)[1]  # 0是前缀，1是格式后缀
            tif_listname.append(name)  # 添加那个文件进入到list name里边
    # 这里到下边分别是打开第一个栅格 然后看长和宽 然后定义一个三维数组 长宽 高就是tif数量有多少
    in_ds = gdal.Open(tif_listname[0])
    band = in_ds.GetRasterBand(1)
    X = band.XSize  # 列数
    Y = band.YSize  # 行数
    print("这个栅格的列和行大小分别为", X, Y)
    return X,Y

def tif_read(img_array,tif_listname): #处理的函数
    count = 0
    for name in tif_listname:  # 对所有的保存的tif路径进行读取
        dataset = gdal.Open(name)
        print('正在处理的栅格是', name)
        if dataset is None:
            print('无法打开这个 *.tif')
            sys.exit(1)  # 退出
        projection = dataset.GetProjection()  # 投影
        transform = dataset.GetGeoTransform()  # 几何信息
        ''' 栅格图片转成数组,依次读取每个波段，然后转换成数组的每个通道 '''
        array_channel = 0  # 数组从通道从0开始
        srcband = dataset.GetRasterBand(1)  # 获取栅格数据集的波段
        arr = srcband.ReadAsArray()  # 将栅格作为数组读取w
        img_array[:, :, count] = np.float64(arr)  # 写入当前栅格
        count = count + 1  # z +1 升一个维度
    print('成功处理了多少文件: ', count)  # 这里是表示处理了多少个tif文件 也是文件夹内有多少文件


def ask_tif_deal(save,tifname,img,askarray,x,y,how_many,asknumber):
    print("正在进行ask计算")
    file_savepath = save
    dataset = gdal.Open(tifname[0])
    projection = dataset.GetProjection()  # 投影
    transform = dataset.GetGeoTransform()  # 几何信息
    row = y  # 矩阵行数
    columns = x  # 列数
    dim_z = 1  # 通道数
    array_channel = 0  # 数组从通道从0开始
    driver = gdal.GetDriverByName('GTiff')  # 创建驱动
    # 创建文件
    dst_ds = driver.Create(file_savepath, columns, row, dim_z, gdal.GDT_Float64)
    dst_ds.SetGeoTransform(transform)  # 设置几何信息
    dst_ds.SetProjection(projection)  # 设置投影
    # 对24个影像做一个平均值
    list_avg24 = []
    for i in range(how_many):  # 通过这个查找到一个有效的队列开始和结束？
        totalsum = 0.0
        bcount = 0
        for arow in range(row):
            sum = 0.0
            acount = 0
            for acol in range(columns):
                if img[arow, acol, i] >= -1 and img[arow, acol, i] <= 1:
                    sum = sum + img[arow, acol, i]
                    acount = acount + 1
            if acount > 0:  # 保证这个列的有效数值都在里边
                sum = sum / acount
                totalsum = totalsum + sum
                bcount = bcount + 1
        totalsum = totalsum / bcount  # 保证这个行有数值
        list_avg24.append(totalsum)
    # 求好了所有均值
    for i in range(row):  # array[行，列，通道】
        for j in range(columns):
            real_y = []
            null_count = 0
            for tmp in range(how_many):  # 这里如果是nodata 数值大于8个就不拟合了
                if img[i, j, tmp] < -1 or img[i, j, tmp] == None:
                    null_count = null_count + 1
            if null_count >= 8:
                askarray[i, j, 0] = -1
                continue
            for tmp in range(how_many):
                if (img[i, j, tmp] < -1 or img[i, j, tmp] > 1 or img[i, j, tmp] == None or img[i, j, tmp] != img[
                    i, j, tmp]):
                    real_y.append(list_avg24[tmp])
                else:
                    real_y.append(img[i, j, tmp])
                    # print(img[i,j,tmp])
            # print(real_y)
            x = np.linspace(1, 24, 24)  # 这里是1开始，24结束，然后是24个文件 按照有多少文件修改后边数字就可以 或者全改成tif_count
            y = np.array(smooth(real_y, 5))  # 这里是sg滤波
            mu = np.mean(y)  # 均值
            sigma = np.std(y)  # sigma
            A2, B2, Sigma2 = curve_fit(f_gauss, x, y)[0]  # 求出高斯拟合参数
            x2 = np.arange(0, 24, 0.07)  # 这里0.07因为是如果 日期选择360的话 那么高斯的协方差公式无法求出，因此24-360是15倍，就是1/15=0.07 从而求出来天数
            y2 = A2 * np.exp(-(x2 - B2) ** 2 / (2 * Sigma2 ** 2))  # 求出对应的高斯拟合y数值
            y2list = y2.tolist()
            ask_day = index_number(y2list,asknumber)
            askarray[i, j, 0] = ask_day
            # print("天数",max_day,"day0",tmp)
        # print('?')

    print("ask天数计算完成")
    map = askarray[:, :, 0]
    dst_ds.GetRasterBand(1).WriteArray(map)
    dst_ds.FlushCache()  # 写入硬盘
    dst_ds = None

    
def max_rate_change(save,tifname,img,maxarray,x,y,how_many):
    print("正在计算最大变化斜率")
    file_savepath = save
    dataset = gdal.Open(tifname[0])
    projection = dataset.GetProjection()  # 投影
    transform = dataset.GetGeoTransform()  # 几何信息
    row = y  # 矩阵行数
    columns = x  # 列数
    dim_z = 1  # 通道数
    array_channel = 0  # 数组从通道从0开始
    driver = gdal.GetDriverByName('GTiff')  # 创建驱动
    # 创建文件
    dst_ds = driver.Create(file_savepath, columns, row, dim_z, gdal.GDT_Float64)
    dst_ds.SetGeoTransform(transform)  # 设置几何信息
    dst_ds.SetProjection(projection)  # 设置投影
    # 对24个影像做一个平均值
    list_avg24 = []
    for i in range(how_many):  
        totalsum = 0.0
        bcount = 0
        for arow in range(row):
            sum = 0.0
            acount = 0
            for acol in range(columns):
                if img[arow, acol, i] >= -1 and img[arow, acol, i] <= 1:
                    sum = sum + img[arow, acol, i]
                    acount = acount + 1
            if acount > 0:  # 保证这个列的有效数值都在里边
                sum = sum / acount
                totalsum = totalsum + sum
                bcount = bcount + 1
        totalsum = totalsum / bcount  # 保证这个行有数值
        list_avg24.append(totalsum)
        # print(totalsum)
    # 求好了所有均值
    for i in range(row):  # array[行，列，通道】
        for j in range(columns):
            real_y = []
            null_count = 0
            for tmp in range(how_many):  
                if img[i, j, tmp] < -1 or img[i, j, tmp] == None:
                    null_count = null_count + 1
            if null_count >= 8:
                maxarray[i, j, 0] = -1
                continue
            for tmp in range(how_many):
                if (img[i, j, tmp] < -1 or img[i, j, tmp] > 1 or img[i, j, tmp] == None or img[i, j, tmp] != img[
                    i, j, tmp]):
                    real_y.append(list_avg24[tmp])
                else:
                    real_y.append(img[i, j, tmp])
            x = np.linspace(1, 24, 24) 
            y = np.array(smooth(real_y, 5))  
            mu = np.mean(y)  # 均值
            sigma = np.std(y)  # sigma
            A2, B2, Sigma2 = curve_fit(f_gauss, x, y)[0]  
            x2 = np.arange(0, 24, 0.07)  
            y2 = A2 * np.exp(-(x2 - B2) ** 2 / (2 * Sigma2 ** 2)) 

            max_rate_day = max_rate_number(y2list)
            maxarray[i, j, 0] = max_rate_day
    print("变化率计算完成")
    map = maxarray[:, :, 0]
    dst_ds.GetRasterBand(1).WriteArray(map)
    dst_ds.FlushCache()  # 写入硬盘
    dst_ds = None

def text_save(filename,data):
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i].replace('[','').replace(']',''))
'''
主函数部分
'''
file_path="" #确定工作空间
file_list=os.listdir(file_path) #便利读取下边所有的
ask=input("获取的ndvi是多少:")

write_text=0
write_filename="log_text.xlsx"
tif_listname=[] #保存的是tif路径
X ,Y= read_filename(file_list,tif_listname)
length = len(tif_listname)#记录有多少个tif文件 这里是根据文件夹内有多少文件来的

img_array = np.zeros((Y, X, length), dtype=np.float64)#设定一个Y行 X列 Z长的三位float数组
tif_count = len(tif_listname)#记录处理了多少文件和表示z轴所在位置
tif_read(img_array,tif_listname)#把tif都提取出来

rate_save_path = ''
max_rate_array = np.zeros((Y, X,1), dtype=np.float64)
max_rate_change(rate_save_path,tif_listname,img_array,max_rate_array,X,Y,tif_count)
