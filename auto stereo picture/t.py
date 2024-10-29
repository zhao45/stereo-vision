import cv2
import threading
import time
import os
import winsound  # For Windows systems

# Create two directories to save the images
save_dir_cam1 = "captured_images/camera_1"
save_dir_cam2 = "captured_images/camera_2"

os.makedirs(save_dir_cam1, exist_ok=True)
os.makedirs(save_dir_cam2, exist_ok=True)

def countdown_and_sound():
    # Countdown 3, 2, 1 and play a beep sound each time
    for i in range(3, 0, -1):
        print(i)
        winsound.Beep(1000, 500)  # Play sound with 1000 Hz frequency for 500 ms
        time.sleep(1)

def capture_image(camera_index, interval, camera_name, save_dir):
    # Open the camera
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Cannot open camera {camera_index}")
        return
    
    counter = 0
    
    while True:
        # Countdown 3, 2, 1 before taking the picture
        countdown_and_sound()

        ret, frame = cap.read()
        
        if not ret:
            print(f"Failed to capture frame from camera {camera_index}")
            break
        
        # Display the camera feed
        cv2.imshow(f"Camera {camera_index}", frame)
        
        # Save the image
        img_name = f"{camera_name}_{counter}.jpg"
        img_path = os.path.join(save_dir, img_name)
        cv2.imwrite(img_path, frame)
        print(f"Image {img_name} saved to {save_dir}")

        counter += 1
        time.sleep(interval)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Set the indexes for the two cameras (0 and 1)
camera_1_index = 0
camera_2_index = 1

# Set the interval for taking pictures (in seconds)
interval = 3

# Use multithreading to control both cameras and save images to different folders
thread_1 = threading.Thread(target=capture_image, args=(camera_1_index, interval, "camera_1", save_dir_cam1))
thread_2 = threading.Thread(target=capture_image, args=(camera_2_index, interval, "camera_2", save_dir_cam2))

# Start the two threads
thread_1.start()
thread_2.start()

# Wait for both threads to finish
thread_1.join()
thread_2.join()
