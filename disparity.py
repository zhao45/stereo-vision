import cv2
import numpy as np

# 初始化兩個攝像頭
cam_left = cv2.VideoCapture(0)  # 左攝像頭
cam_right = cv2.VideoCapture(1)  # 右攝像頭

while True:
    # 獲取左右攝像頭的畫面
    ret_left, frame_left = cam_left.read()
    ret_right, frame_right = cam_right.read()

    if not ret_left or not ret_right:
        print("無法讀取攝像頭畫面")
        break

    # 調整圖像大小使兩者一致
    frame_right = cv2.resize(frame_right, (frame_left.shape[1], frame_left.shape[0]))

    # 將左邊圖像轉換為紅色通道
    red_channel = np.zeros_like(frame_left)
    red_channel[:, :, 2] = frame_left[:, :, 2]

    # 將右邊圖像轉換為藍色通道
    blue_channel = np.zeros_like(frame_right)
    blue_channel[:, :, 0] = frame_right[:, :, 0]

    # 合併兩張圖像生成紅藍立體效果圖
    stereo_image = cv2.addWeighted(red_channel, 0.5, blue_channel, 0.5, 0)

    # 顯示紅藍立體圖
    cv2.imshow('Red-Blue Stereo Vision', stereo_image)

    # 按 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放攝像頭並關閉視窗
cam_left.release()
cam_right.release()
cv2.destroyAllWindows()
