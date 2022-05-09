import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

#参数设置
PI = math.pi
lamda = 6.50e-4
k = 2*PI/lamda
sphd = 6.36e-3
spod = 0.1
w_d_num = math.ceil(spod/sphd)#波前记录平main对应向点间距数量

zo = 300
zw = 10
zwo = zo - zw
zw_1 = 1/zw

w_area = abs(zw)*(lamda/2/sphd)
Lw = w_area*2/math.sqrt(2)
rw = math.ceil(Lw/sphd)#一个波前的像素数
cw = rw
Lw = rw*sphd
xw = np.arange(-Lw/2, Lw/2, sphd)
yw = np.arange(-Lw/2, Lw/2, sphd)
xw, yw = np.meshgrid(xw, yw)

#读取图像数据
I0 = cv.imread(r"pic\z.jpg")
# I0 = uint8(I0);
I0_b,I0_g,I0_r = cv.split(I0)
# cv.imwrite(r'pic\z2.jpg', I0_r)
r_I0, c_I0, d_I0 = I0.shape
I0_r_UF = I0_r*math.e**(1j*np.random.rand(r_I0, c_I0)*2*PI)

#设置全息面
r = 3000
c = 3000
Lhx = sphd*r
Lhy = sphd*c
xh = np.arange(-Lhx/2, Lhx/2, sphd)
yh = np.arange(-Lhy/2, Lhy/2, sphd)
xh, yh = np.meshgrid(xh, yh)

#设置波前记录平面
x_wrp = xh
y_wrp = yh
#一个点的波前
I0_F0 = math.e**(1j*k*zw)/(1j*lamda*zw)
I0_F1 = math.e**(1j*k/2/zw*(xw**2+yw*yw))
UF0_wrp = I0_F0*I0_F1

#wrp上物光波起始位置
wrp_rl = round(( r-(rw+(r_I0-1)*w_d_num) ) / 2 )
wrp_cl = wrp_rl

UF_wO = np.zeros([r, c])*1j

#计算wrp上物光波
for i_o in range(0, r_I0):
    i_o_rl = wrp_rl + i_o * w_d_num
    i_o_rh = i_o_rl + rw
    for j_o in range(0, c_I0):
        j_o_rl = wrp_cl + j_o * w_d_num
        j_o_rh = j_o_rl + cw

        UF_I0 = (I0_r_UF[i_o][j_o]) * UF0_wrp
        UF_wO[i_o_rl: i_o_rh, j_o_rl: j_o_rh] = UF_I0

#------------------TFFT----------------------------
# F0 = math.e**(1j*k*zwo)/(1j*lamda*zwo)
# F1 = math.e**(1j*k/2/zwo*(xh**2+yh*yh))
#
# fF1 = np.fft.fft2(F1)
# fI0 = np.fft.fft2(UF_wO)
# fu = fF1*fI0
# O = F0 * np.fft.fftshift(np.fft.ifft2(fu))

#------------------DFFT----------------------------
uvd = 1/Lhx
u_xo = np.arange(-r/Lhx/2, r/Lhx/2, uvd )
v_yo = np.arange(-c/Lhy/2, c/Lhy/2, uvd)
u_xo, v_yo = np.meshgrid(u_xo, v_yo)
# F = math.e**(1j)
F0 = math.e**(1j*k*zwo*( 1-lamda*lamda*(u_xo**2+v_yo**2)/2 ))
fF1 = np.fft.fft2(UF_wO)
O = fF1*F0
O = np.fft.ifft2(O)

R = k*( xh*np.sin(np.deg2rad(0)) + yh*np.sin(np.deg2rad(8)) )
IT = np.abs(O) * np.cos(np.angle(O)-R)

IT = (IT - np.min(IT))/( np.max(IT) - np.min(IT) )*255
cv.imwrite(r'pic\2.bmp',abs(IT))

