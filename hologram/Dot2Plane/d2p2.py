import math
import cv2 as cv
import numpy as np
import torch as tr
# import minpy.numpy as np
#用于输入输出mat
#import scipy.io as scio
# from matplotlib import pyplot as plt
import time

# tr.cuda.is_available()
# tr.cuda.device_count()
#基础参数设置
PI = math.pi
spd = 0.1
sphd = 3.18e-4
SRGB_TH = 50
z = 300
rh = 5000
ch = 5000

Lxh = sphd*ch
Lyh = sphd*rh

#图像读取
Img = cv.imread(r"F:\python\code\hologram\pic\z.jpg")
# I0 = uint8(I0);
I0_b, I0_g, I0_r = cv.split(Img)
rI0, cI0, dI0 = Img.shape

I0 = [I0_r, I0_g, I0_b]
lamda = [6.50e-4, 5.20e-4,4.50e-4]
yh_shift3 = [-65, -50, -35]

#选择图片通道数
for i_depth in range(1):
    lamda_i = lamda[i_depth]
    k = 2*PI/lamda_i
    # a = lamda[i_depth]
    # k = 2*PI/a
    I0_squre = I0[i_depth] * math.e**(1j*np.random.rand(rI0,cI0)*2*PI)
    Amp = np.zeros((rI0*cI0,1))*1j
    xyDis = np.zeros((rI0*cI0,2))
    I0_2 = np.zeros((rI0,cI0))

    xmo = cI0&1
    ymo = rI0&1
    xy_num =0

#选择图像有效灰度和位置信息
    for i_object in range(rI0):
        y_object = -(rI0-ymo)/2*spd + i_object*spd
        for j_object in range(cI0):
            if I0[i_depth][i_object,j_object] > SRGB_TH:
                I0_2[i_object,j_object] = I0_g[i_object,j_object]
                x_object = -(cI0 - xmo) / 2 * spd + j_object * spd
                Amp[xy_num] = I0_squre[i_object,j_object]
                xyDis[xy_num, 0] = x_object
                xyDis[xy_num, 1] = y_object
                xy_num = xy_num + 1

    O_AMP = Amp[:xy_num]
    O_XYDIS = xyDis[:xy_num, :]

    xh = np.arange(-Lxh / 2, Lxh / 2, sphd)
    xh = xh + yh_shift3[i_depth]
    yh = np.arange(-Lyh / 2, Lyh / 2, sphd)
    xh, yh = np.meshgrid(xh, yh)

    expjkz_jlz = math.e**(1j*k*z)/1j/lamda_i/z
    jk_2z = 1j*k/z/2

    UF1 = np.zeros((rh, ch))*1j
    UF = np.zeros((rh, ch))*1j
    
    #numpy转为torch中GPU数据
    # expjkz_jlz_tr = tr.from_numpy(expjkz_jlz)
    # jk_2z_tr = tr.from_numpy(jk_2z)
    UF_tr = tr.from_numpy(UF).cuda()
    UF1_tr = tr.from_numpy(UF1).cuda()
    O_XYDIS_tr = tr.from_numpy(O_XYDIS).cuda()
    O_AMP_tr = tr.from_numpy(O_AMP).cuda()
    xh_tr = tr.from_numpy(xh).cuda()
    yh_tr = tr.from_numpy(yh).cuda()
    
    #计算物光波
    for o_num in range(xy_num):
        start = time.perf_counter()

        # xo = O_XYDIS[o_num, 0]
        # yo = O_XYDIS[o_num, 1]
        xo = O_XYDIS_tr[o_num, 0]
        yo = O_XYDIS_tr[o_num, 1]
        
        # UF1 = O_AMP[o_num, 0] * expjkz_jlz * math.e ** (jk_2z * ((xh - xo) ** 2 + (yh - yo) ** 2))
        UF1_tr = O_AMP_tr[o_num, 0] * expjkz_jlz * math.e ** (jk_2z * ((xh_tr - xo) ** 2 + (yh_tr - yo) ** 2))
        # UF = UF1 + UF
        UF_tr = UF1_tr + UF_tr
        

        end = time.perf_counter()
        print('Running time: %s Seconds' % (end - start))

    UF = UF_tr.cpu()
    UF = UF.numpy()
    # R = math.e ** (1j * k * (xh * np.sin(np.deg2rad(0)) + yh * np.sin(np.deg2rad(0))))
    R = k * (xh * np.sin(np.deg2rad(0)) + yh * np.sin(np.deg2rad(0)))
    IT = np.abs(UF) * np.cos(np.angle(UF) - R)
    IT = (IT - np.min(IT)) / (np.max(IT) - np.min(IT)) * 255
    cv.imwrite(r'pic\1.bmp', abs(IT))



print(xmo,ymo)
