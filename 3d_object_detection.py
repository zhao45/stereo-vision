import cv2

# 初始化攝像頭
cam_left = cv2.VideoCapture(0)  # 左攝像頭
cam_right = cv2.VideoCapture(1)  # 右攝像頭

# 使用預訓練的人體檢測模型 (OpenCV DNN 中的 MobileNetSSD)
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

while True:
    # 獲取左右攝像頭的畫面
    ret_left, frame_left = cam_left.read()
    ret_right, frame_right = cam_right.read()

    if not ret_left or not ret_right:
        print("無法讀取攝像頭畫面")
        break

    # 將圖像尺寸縮放至模型要求的尺寸
    h, w = frame_left.shape[:2]
    blob_left = cv2.dnn.blobFromImage(cv2.resize(frame_left, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    blob_right = cv2.dnn.blobFromImage(cv2.resize(frame_right, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # 檢測左邊畫面中的人
    net.setInput(blob_left)
    detections_left = net.forward()

    # 檢測右邊畫面中的人
    net.setInput(blob_right)
    detections_right = net.forward()

    # 在左畫面上繪製邊框
    for i in range(detections_left.shape[2]):
        confidence = detections_left[0, 0, i, 2]
        if confidence > 0.5:  # 設定置信度閾值
            box = detections_left[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame_left, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # 在右畫面上繪製邊框
    for i in range(detections_right.shape[2]):
        confidence = detections_right[0, 0, i, 2]
        if confidence > 0.5:
            box = detections_right[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame_right, (startX, startY), (endX, endY), (255, 0, 0), 2)

    # 結合畫面（左右並排）
    combined_frame = cv2.hconcat([frame_left, frame_right])

    # 顯示結果
    cv2.imshow("3D Object Detection - Human", combined_frame)

    # 按 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放攝像頭並關閉視窗
cam_left.release()
cam_right.release()
cv2.destroyAllWindows()
