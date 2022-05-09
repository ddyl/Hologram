import time
import math
import cv2
import cv2 as cv
import numpy as np
import zipfile
import scipy.io as scio
from matplotlib import pyplot as plt

start = time.perf_counter()

I0 = cv.imread(r"pic\man.png")
# I0 = uint8(I0);
I0_b, I0_g, I0_r = cv.split(I0)
I0_gray = cv.cvtColor(I0, cv2.COLOR_BGR2GRAY)
# plt.imshow(abs(I0_gray),cmap='gray')
# plt.show()

I0_rgb = cv.merge([I0_r, I0_g, I0_b])
rI0, cI0, dI0 = I0.shape
rI0_int = rI0/2
cI0_int = cI0/2

#显示数据装入零矩阵
r = 1000
c = 1000
r_padl = int(r/2 - rI0/2)
c_padl = int(c/2 - cI0/2)
r_padh = int(r/2 + rI0/2)
# c_padh = int(c/2 + cI0/2)
c_padh = c/2 + cI0/2
c_padh = int(c_padh)

I0_squre = np.zeros((r,c))
I0_squre[r_padl:r_padh, c_padl:c_padh] = I0_gray#在矩阵写入影像矩阵
# plt.imshow(I0_squre)
# plt.show()

PI = math.pi
lamda = 6.50e-4
k = 2*PI/lamda
I0_squre = I0_squre * math.e**(1j*np.random.rand(r,c)*2*PI)

z0 = 300
sphd = 3.18e-4
Lhx = sphd*r
Lhy = sphd*c
xh = np.arange(-Lhx/2, Lhx/2-sphd*0.001, sphd)
yh = np.arange(-Lhy/2, Lhy/2-sphd*0.001, sphd)
xh, yh = np.meshgrid(xh, yh)

Lox = r*lamda*z0/Lhx
Loy = c*lamda*z0/Lhx
# spod = Lox/r
spod = lamda*z0/r/sphd
xo = np.arange(-Lox/2, Lox/2-spod*0.001, spod)
yo = np.arange(-Loy/2, Loy/2-spod*0.001, spod)
xo, yo = np.meshgrid(xo,yo)


# F = math.e**(1j)
F0 = math.e**(1j*k*z0)/(1j*lamda*z0) * math.e**(1j*k/2/z0*(xh**2+yh*yh))
F = math.e**(1j*k/2/z0*(xo**2+yo*yo))
# print(xo**2)
fF1 = np.fft.fft2(np.fft.fftshift(I0_squre * F))
O = F0*fF1

#保存与读取物光波复数矩阵
np.save("o.npy",O)
O1=np.load("o.npy")

#scipy只能保存字典格式,给matlab使用
scio.savemat('a.mat',{'data':O})
O1 = scio.loadmat('a.mat')
O1_data = O1['data']

#下面方法无法保存复数矩阵
# np.savetxt("a.txt",O)
# O1 = np.loadtxt("a.txt")


#压缩文件
# try to change compression type
try:
    import zlib
    compression = zipfile.ZIP_DEFLATED
except:
    compression = zipfile.ZIP_STORED
modes = {zipfile.ZIP_DEFLATED: 'deflated', zipfile.ZIP_STORED: 'stored'}

zf = zipfile.ZipFile('a.zip',mode='w')
try:
    zf.write('o.npy',compress_type=compression)
finally:
    zf.close()

R = math.e**( 1j*k*( xh*np.sin(np.deg2rad(0)) + yh*np.sin(np.deg2rad(15)) ) )
inter = (O1 - np.min(abs(O1)))/(np.max(np.abs(O1)) - np.min(abs(O1))) + R
IT = inter*inter.conjugate()
IT = (IT - np.min(IT))/(np.max(IT) - np.min(IT) )*255

# plt.imshow(abs(IT),cmap='gray')
# plt.show()
cv.imwrite(r'pic\1.bmp',abs(IT))

end = time.perf_counter()

print('Running time: %s Seconds'%(end-start))