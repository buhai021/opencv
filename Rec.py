#先通过边缘检测分辨颜色，再通过轮廓检测分辨形状
import cv2                      #opencv库
import numpy as np
import matplotlib.pyplot as plt

#图片预处理
img=cv2.imread('image7.jpg')                        #读取图片
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)       #将图片从bgr转为灰度图
img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

area_image=img.copy()
length_image=img.copy()
#根据hsv框定颜色范围
lower_red=np.array([0,100,100])
upper_red=np.array([10,255,255])
lower_blue=np.array([110,100,100])
upper_blue=np.array([130,255,255])
lower_green=np.array([50,100,100])
upper_green=np.array([70,255,255])

def cable_detection(photo,color='red',debug=False):
    red_mask=cv2.inRange(photo,lower_red,upper_red)
    blue_mask=cv2.inRange(photo,lower_blue,upper_blue)
    green_mask=cv2.inRange(photo,lower_green,upper_green)

    kernel=np.ones((5,5),np.uint8)
    red_mask=cv2.morphologyEx(red_mask,cv2.MORPH_OPEN,kernel)
    blue_mask=cv2.morphologyEx(blue_mask,cv2.MORPH_OPEN,kernel)
    green_mask=cv2.morphologyEx(green_mask,cv2.MORPH_OPEN,kernel)

    contours,_=cv2.findContours(red_mask+blue_mask+green_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x,y,w,h=cv2.boundingRect(contour)
        color=''
        if cv2.contourArea(contour)>100:
            if np.any(red_mask[y:y+h,x:x+w]):
                color = "红色"
                cv2.rectangle(photo,(x,y),(x+w,y+h),(0,0,255),2)
            elif np.any(blue_mask[y:y+h,x:x+w]):
                color = "蓝色"
                cv2.rectangle(photo,(x,y),(x+w,y+h),(255,0,0),2)
            elif np.any(green_mask[y:y+h,x:x+w]):
                color = "绿色"
                cv2.rectangle(photo,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(photo,color,(x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,0.9, (0, 255, 0), 2)

    return photo
cable_detection(img_hsv)
cv2.imshow('cable_detection',img_hsv)

 #梯度运算提取图片的边缘

 #canny算子
image_canny80_150=cv2.Canny(img_gray,80,150)
image_canny100_200=cv2.Canny(img_gray,100,200)
plt.subplot(131),plt.imshow(image_canny80_150),plt.title('Canny50_100'),plt.axis('off')
plt.subplot(132),plt.imshow(image_canny100_200),plt.title('Canny100_200'),plt.axis('off')
plt.subplot(133),plt.imshow(img_gray),plt.title('Gray'),plt.axis('off')
plt.show()
#cv2.imshow('80_150',image_canny80_150)
#cv2.imshow('100_200',image_canny100_200)
#cv2.imshow('morph',morph)


#sobel算子
#分别对x，y方向求导得出xy方向的边缘
sobel_x=cv2.Sobel(img_gray,ddepth=cv2.CV_64F,dx=1,dy=0,ksize=3) #64位双精度浮点数保留负值，3*3卷积核
sobel_y=cv2.Sobel(img_gray,ddepth=cv2.CV_64F,dx=0,dy=1,ksize=3)
sobel_magnitude=cv2.magnitude(sobel_x,sobel_y)                  #融合两个边缘图算出强弱的边缘
sobel_magnitude_Abs=cv2.convertScaleAbs(sobel_magnitude)        #计算绝对值,结果转换为CV_8U
#1行2列，黑白图，标题，不显示坐标轴
plt.subplot(121),plt.imshow(sobel_magnitude_Abs, cmap='gray'),plt.title('sobel_magnitude_Abs'),plt.axis('off')
plt.show()
'''
#scharr算子
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
#提取轮廓
_,threshold_image=cv2.threshold(image_canny80_150,127,255,cv2.THRESH_BINARY)#忽略实际使用的阈值
contours,_=cv2.findContours(threshold_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#建立轮廓等级树，仅保留方向终点的坐标

for cnt in contours:
    area = cv2.contourArea(cnt)
    length = cv2.arcLength(cnt,not True)
    print(area)
    #画出较大轮廓
    #if 100<area<10000:  #面积不可用，因为很难能确定得出闭合曲线
    if 80<length<1000:
        cv2.drawContours(area_image,[cnt],-1,(0,0,255),2)
        cv2.drawContours(length_image,[cnt],-1,(0,255,0),2)
        #计算轮廓的质心，即所有像素平均位置
        retval=cv2.moments(cnt)
        if retval['m00']!=0:#零阶矩，表示像素总数
            #一阶矩:所有像素在x方向的位置乘以其强度后的总和。
            cx=int(retval['m10']/retval['m00'])#计算x方向加权平均位置
            cy=int(retval['m01']/retval['m00'])#计算y方向加权平均位置
        else:
            cx=0
            cy=0
        #绘制周长和面积
        cv2.putText(area_image,text=f'{area:.2f}',org=(cx,cy+5),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(0,0,0),thickness=2)
        cv2.putText(length_image,text=f'{length}',org=(cx,cy+5),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.5,color=(0,0,0),thickness=2)

plt.subplot(221),plt.imshow(threshold_image, cmap='gray'),plt.title('Threshold Image'),plt.axis('off')
plt.subplot(222),plt.imshow(area_image,cmap='gray'),plt.title('contours'),plt.axis('off')
plt.subplot(223),plt.imshow(length_image,cmap='gray'),plt.title('length'),plt.axis('off')
plt.show()
cv2.imshow('threshold_image',threshold_image)
cv2.imshow('area_image',area_image)
cv2.imshow('length_image',length_image)


cv2.waitKey(0)                                  #等待任意输入进入下一步
cv2.destroyAllWindows()                         #清除显示窗口
