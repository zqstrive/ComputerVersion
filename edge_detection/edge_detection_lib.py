import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lena.jpg')
lena_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# 灰度化处理图像
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# 高斯滤波
gauss_blur = cv2.GaussianBlur(gray_img,(3,3),0)

# 阈值处理
ret, binary = cv2.threshold(gauss_blur,127,255,cv2.THRESH_BINARY)

# roberts算子
kernel_x = np.array([[-1,0],[0,1]],dtype=int)
kernel_y = np.array([[0,-1],[1,0]],dtype=int)
x = cv2.filter2D(binary, cv2.CV_16S, kernel_x)
y = cv2.filter2D(binary, cv2.CV_16S, kernel_y)
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

#Prewitt算子
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]], dtype=int)
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=int)
x = cv2.filter2D(binary, cv2.CV_16S, kernelx)
y = cv2.filter2D(binary, cv2.CV_16S, kernely)
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
Prewitt = cv2.addWeighted(absX,0.5,absY,0.5,0)

#Sobel算子
x = cv2.Sobel(binary, cv2.CV_16S, 1, 0)
y = cv2.Sobel(binary, cv2.CV_16S, 0, 1)
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

#Canny算子
Canny = cv2.Canny(binary,127,255)
Canny = cv2.convertScaleAbs(Canny)

titles = ['Source Image', 'Binary Image', 'Roberts Image',
          'Prewitt Image','Sobel Image', 'Canny Image']
images = [lena_img, binary, Roberts, Prewitt, Sobel, Canny]
for i in np.arange(6):
   plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
   plt.title(titles[i])
   plt.xticks([]),plt.yticks([])
plt.show()
