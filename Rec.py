#先通过边缘检测分辨颜色，再通过轮廓检测分辨形状
import cv2                      #opencv库
import matplotlib.pyplot as plt
#图片预处理
img=cv2.imread('image1.jpg')                        #读取图片
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)           #将图片从bgr转为灰度图

#梯度运算提取图片的边缘
#sobel算子
#分别对x，y方向求导得出xy方向的边缘
sobel_x=cv2.Sobel(img_gray,ddepth=cv2.CV_64F,dx=1,dy=0,ksize=3) #64位双精度浮点数保留负值，3*3卷积核
sobel_y=cv2.Sobel(img_gray,ddepth=cv2.CV_64F,dx=0,dy=1,ksize=3)
sobel_magnitude=cv2.magnitude(sobel_x,sobel_y)                  #融合两个边缘图算出强弱的边缘
sobel_magnitude_Abs=cv2.convertScaleAbs(sobel_magnitude)        #计算绝对值,结果转换为CV_8U
#1行2列，黑白图，标题，不显示坐标轴
plt.subplot(121),plt.imshow(img_gray, cmap='gray'),plt.title('Original Image'),plt.axis('off')
plt.subplot(122),plt.imshow(sobel_magnitude_Abs, cmap='gray'),plt.title('sobel_magnitude_Abs'),plt.axis('off')
plt.show()
'''#scharr算子
scharr_x=cv2.Scharr(img_gray, ddepth=cv2.CV_64F,dx=1,dy=0)
scharr_y=cv2.Scharr(img_gray, ddepth=cv2.CV_64F,dx=0,dy=1)
scharr_magnitude=cv2.magnitude(scharr_x,scharr_y)
scharr_magnitude_Abs=cv2.convertScaleAbs(scharr_magnitude)
cv2.imshow('scharr_magnitude',scharr_magnitude)
cv2.imshow('scharr_magnitude_Abs',scharr_magnitude_Abs)
plt.subplot(121),plt.imshow(img_gray, cmap='gray'),plt.title('Original Image'),plt.axis('off')
plt.subplot(122),plt.imshow(scharr_magnitude_Abs, cmap='gray'),plt.title('scharr_magnitude_Abs'),plt.axis('off')
plt.show()
'''
cv2.waitKey(0)                                  #等待任意输入进入下一步
cv2.destroyAllWindows()                         #清除显示窗口
