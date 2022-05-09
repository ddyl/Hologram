import time


import math
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

start = time.perf_counter()

'''图像数据处理'''
I0 = cv.imread(r"pic\z.jpg")
# I0 = uint8(I0);
I0_b,I0_g,I0_r = cv.split(I0)
I0_rgb = cv.merge([I0_r, I0_g, I0_b])
rI0, cI0, dI0 = I0.shape
rI0_int = rI0/2
cI0_int = cI0/2

r = 2000
c = 2000
r_padl = int(r/2 - rI0/2)
c_padl = int(c/2 - cI0/2)
r_padh = int(r/2 + rI0/2)
# c_padh = int(c/2 + cI0/2)
c_padh = c/2 + cI0/2
c_padh = int(c_padh)

I0_squre = np.zeros((r,c))
I0_squre[r_padl:r_padh, c_padl:c_padh] = I0_r
# plt.imshow(I0_squre)
# plt.show()

'''DFFT运算过程'''
PI = math.pi
lamda = 6.50e-4
k = 2*PI/lamda
I0_squre = I0_squre * math.e**(1j*np.random.rand(r,c)*2*PI)#随机相位

z0 = 300
sphd = 6.36e-3
Lhx = sphd*r
Lhy = sphd*c
xh = np.arange(-Lhx/2, Lhx/2, sphd)
yh = np.arange(-Lhy/2, Lhy/2, sphd)
xh, yh = np.meshgrid(xh, yh)

uvd = 1/Lhx

u_xo = np.arange(-r/Lhx/2, r/Lhx/2, uvd )
v_yo = np.arange(-c/Lhy/2, c/Lhy/2, uvd)
u_xo, v_yo = np.meshgrid(u_xo, v_yo)


# F = math.e**(1j)
F0 = math.e**(1j*k*z0*( 1-lamda*lamda*(u_xo**2+v_yo**2)/2 ))

# print(xo**2)
# fF1 = np.fft.fftshift(np.fft.fft2(I0_squre))
fF1 = np.fft.fft2(I0_squre)
O = fF1*F0
O = np.fft.ifft2(O)

R = math.e**( 1j*k*( xh*np.sin(np.deg2rad(0)) + yh*np.sin(np.deg2rad(0)) ) )
inter = (O - np.min(abs(O)))/(np.max(np.abs(O)) - np.min(abs(O))) + R
IT = inter*inter.conjugate()
IT = (IT - np.min(IT))/(np.max(IT) - np.min(IT) )*255

plt.imshow(abs(IT),cmap='gray')



cv.imwrite(r'pic\2.bmp',abs(IT))

end = time.perf_counter()

print('Running time: %s Seconds'%(end-start))