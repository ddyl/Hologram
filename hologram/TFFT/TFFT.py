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

r = 1000
c = 1000
r_padl = int(r/2 - rI0/2)
c_padl = int(c/2 - cI0/2)
r_padh = int(r/2 + rI0/2)
# c_padh = int(c/2 + cI0/2)
c_padh = c/2 + cI0/2
c_padh = int(c_padh)

I0_squre = np.zeros((r,c))
I0_squre[r_padl:r_padh, c_padl:c_padh] = I0_r

#参数设置
PI = math.pi
lamda = 6.50e-4
k = 2*PI/lamda
I0_squre = I0_squre*math.e**(1j*np.random.rand(r,c)*2*PI)

z0 = 300
sphd = 8e-3
Lox = sphd*r
Loy = sphd*c

xo = np.arange(-Lox/2, Lox/2-sphd*0.001, sphd)
yo = np.arange(-Loy/2, Loy/2-sphd*0.001, sphd)
xo,yo = np.meshgrid(xo,yo)

spod = lamda*z0/r/sphd
# =============================================================================
# xo = tr.arange(-Lox/2, Lox/2-spdh, spdh)
# yo = tr.arange(-Loy/2, Loy/2-spdh, spdh)
# xo,yo = tr.meshgrid(xo,yo)
# =============================================================================


# F = math.e**(1j)
F0 = math.e**(1j*k*z0)/(1j*lamda*z0)
F1 = math.e**(1j*k/2/z0*(xo**2+yo*yo))
# print(xo**2)
fF1 = np.fft.fft2(F1)
fI0 = np.fft.fft2(I0_squre)

fu = fF1*fI0
# O = F0 * np.fft.ifftshift(np.fft.ifft2(fu))
O = F0 * np.fft.fftshift(np.fft.ifft2(fu))

# R = math.e**(1j*k*(xo*np.sin(np.deg2rad(0)) + yo*np.sin(np.deg2rad(15)) ))
# R = math.exp( 1j*k*( xo*np.sin(np.deg2rad(0)) + yo*np.sin(np.deg2rad(8)) ) )
# inter = O/np.max(np.abs(O)) + tr.ones(r,c).numpy()
# inter = O/np.max(np.abs(O)) + np.ones((r,c))
# inter = O/np.max(np.abs(O)) + R
# IT = inter*inter.conjugate()

R = k*( xo*np.sin(np.deg2rad(0)) + yo*np.sin(np.deg2rad(15)) )
IT = np.abs(O) * np.cos(np.angle(O)-R)

IT = (IT - np.min(IT))/(np.max(IT) - np.min(IT) )*255

plt.imshow(abs(IT),cmap='gray')
plt.imshow(IT.real,cmap='gray')
plt.imshow(IT.imag,cmap='gray')


cv.imwrite(r'pic\1.bmp',abs(IT))
cv.imwrite(r'pic\2.bmp',real(IT))

# plt.subplot(221),plt.imshow(I0_r,cmap='gray')
# plt.subplot(222),plt.imshow(I0_g,cmap='gray')
# plt.subplot(223),plt.imshow(I0_b,cmap='gray')
# plt.subplot(224),plt.imshow(I0_rgb,cmap='gray')

# print(max(1,2,3))
# =============================================================================
# c1 = range(-5, 5, 1)
# print(c1[9])
#
# c = tr.tensor([[1,2,3],
#      [1,2,3],
#      [1,2,3]])
# print(c*c)
#
# cf = np.fft.fft2(c)
# cf_abs = np.abs(cf)
# cf_shift = np.fft.fftshift(cf)
# cf_shift_abs = np.abs(cf_shift)
#
# plt.subplot(121),plt.imshow(c,cmap='gray')
# plt.subplot(122),plt.imshow(cf_shift_abs,cmap='gray')
#
# =============================================================================

end = time.perf_counter()

print('Running time: %s Seconds'%(end-start))