import cv2
import os

def pre_process_image_in_folder(input_dir_path, output_dir_path):
    files = os.listdir(input_dir_path)
    for file in files:
        file_path = os.path.join(input_dir_path, file)
        alpha = 2.5 # Contrast control (1.0-3.0)
        beta = 0 # Brightness control (0-100)
        img = cv2.imread(file_path)
        
        adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)
        greenMask = cv2.inRange(hsv, (26, 10, 30), (97, 100, 255))

        hsv[:,:,1] = greenMask

        output_file_path  =os.path.join( output_dir_path, file)
        back = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite(output_file_path, back)
        