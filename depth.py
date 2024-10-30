import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# 初始化左右摄像头
capL = cv.VideoCapture(0)  # 左摄像头
capR = cv.VideoCapture(1)  # 右摄像头

# 设置立体匹配器
stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)

while True:
    # 从左右摄像头读取帧
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    
    if not retL or not retR:
        print("无法从摄像头读取帧")
        break
    
    # 转换为灰度图像
    grayL = cv.cvtColor(frameL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(frameR, cv.COLOR_BGR2GRAY)

    # 计算视差
    disparity = stereo.compute(grayL, grayR)
    
    # 归一化视差图以便显示
    disp_norm = cv.normalize(disparity, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    
    # 显示视差图
    cv.imshow('Disparity', disp_norm)
    
    # 按下 'q' 键退出
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头和关闭窗口
capL.release()
capR.release()
cv.destroyAllWindows()
