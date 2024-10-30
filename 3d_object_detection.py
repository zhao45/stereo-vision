import cv2
import numpy as np

# 讀取左右鏡頭圖像
left_cam = cv2.VideoCapture(0)   # 左鏡頭
right_cam = cv2.VideoCapture(1)  # 右鏡頭

# 配置立體匹配器
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# 校正和視差計算函數
def calculate_depth_map(left_image, right_image):
    # 轉換為灰度圖
    gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    # 計算視差
    disparity = stereo.compute(gray_left, gray_right)
    depth_map = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return depth_map

# 主循環
while True:
    # 讀取左右圖像
    ret_left, left_frame = left_cam.read()
    ret_right, right_frame = right_cam.read()

    if not ret_left or not ret_right:
        print("無法獲取攝像機畫面")
        break

    # 計算深度圖
    depth_map = calculate_depth_map(left_frame, right_frame)

    # 使用人臉檢測作為示例
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(left_frame, 1.3, 5)

    # 在人臉處標記深度
    for (x, y, w, h) in faces:
        cv2.rectangle(left_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_center = (x + w//2, y + h//2)
        depth = depth_map[face_center[1], face_center[0]]
        
        # 顯示深度信息
        cv2.putText(left_frame, f"Depth: {depth} mm", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 顯示結果
    cv2.imshow("Left Camera", left_frame)
    cv2.imshow("Depth Map", depth_map)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
left_cam.release()
right_cam.release()
cv2.destroyAllWindows()
