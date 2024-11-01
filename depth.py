import cv2
import numpy as np

# 初始化攝像頭
cap_left = cv2.VideoCapture(0)
cap_right = cv2.VideoCapture(1)

# 設定畫面大小
width = 1280
height = 720
cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# 初始化 ORB 特徵檢測器
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 初始化顯示畫面為黑色
display_frame = np.zeros((height, width, 3), dtype=np.uint8)

while True:
    # 捕捉鏡頭畫面
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()
    
    if not ret_left or not ret_right:
        print("無法捕捉鏡頭畫面")
        break

    # 轉為灰階圖像
    gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    # 檢測 ORB 特徵點並計算描述符
    keypoints_left, descriptors_left = orb.detectAndCompute(gray_left, None)
    keypoints_right, descriptors_right = orb.detectAndCompute(gray_right, None)

    # 特徵點匹配
    matches = bf.match(descriptors_left, descriptors_right)
    matches = sorted(matches, key=lambda x: x.distance)

    # 計算重疊區域
    if len(matches) > 10:
        src_pts = np.float32([keypoints_left[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_right[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # 計算單應矩陣（Homography）
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if M is not None:
            # 將右鏡頭圖像變換到左鏡頭的坐標系中
            aligned_right = cv2.warpPerspective(frame_right, M, (width, height))

            # 計算重疊區域
            overlap_region = cv2.bitwise_and(frame_left, aligned_right)

            # 更新顯示畫面
            display_frame = np.zeros_like(frame_left)
            non_zero_indices = np.where(overlap_region != 0)
            display_frame[non_zero_indices[0], non_zero_indices[1]] = overlap_region[non_zero_indices[0], non_zero_indices[1]]
        else:
            print("無法計算單應矩陣，顯示全黑畫面")
            display_frame = np.zeros((height, width, 3), dtype=np.uint8)  # 顯示全黑畫面
    else:
        print("特徵點匹配數量不足，顯示全黑畫面")
        display_frame = np.zeros((height, width, 3), dtype=np.uint8)  # 顯示全黑畫面

    # 顯示重疊畫面
    cv2.imshow("Overlap Region", display_frame)

    # 退出條件
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放攝像頭和關閉窗口
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
