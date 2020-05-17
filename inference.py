import matplotlib.pyplot as plt
from deepforest import deepforest
from deepforest import get_data
from deepforest import utilities
import cv2
import os
import numpy as np
from deepforest import preprocess
from preProcess import pre_process_image_in_folder
test_model = deepforest.deepforest()
test_model.use_release()

#predict image
pre_process_image_in_folder("./data", "./processed")

images = os.listdir('./processed')
for image in images:
    image_path = os.path.join("./processed", image)
    raster = cv2.imread(image_path)
    numpy_image = np.array(raster)
    windows = preprocess.compute_windows(numpy_image, patch_size=400,patch_overlap=0.1)
    for window in windows:
        crop = numpy_image[window.indices()]

        img = test_model.predict_image(numpy_image = crop,return_plot=True, score_threshold=0.05)


        #Show image, matplotlib expects RGB channel order, but keras-retinanet predicts in BGR
        plt.imshow(img[...,::-1])
        plt.show()
    
#raster = cv2.imread("./processed/IITK1.JPG")
#numpy_image = np.array(raster)
#windows = preprocess.compute_windows(numpy_image, patch_size=400,patch_overlap=0.1)
#image = model.predict_image(numpy_image = crop,return_plot=True, score_threshold=0.05)


