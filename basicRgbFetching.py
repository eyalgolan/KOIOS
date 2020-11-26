import cv2
import numpy as np
import scipy
import sklearn.decomposition
import matplotlib.pyplot as plt
import cv2.cv2
import sklearn.datasets

# rgb_image = cv2.imread('Photo_of_me.jpg')
#
# red,green,blue = cv2.split(rgb_image)
# cv2.imshow('G-RGB', green)
# cv2.waitKey(0)
#

# This part convert the video to images, every image is a frame
vidcap = cv2.VideoCapture('movie.mp4')
success, image = vidcap.read()
num_of_frames = 0
average = []
fps = vidcap.get(cv2.CAP_PROP_FPS)
blues = []
reds = []
while success:
    red, green, blue = cv2.split(image)
    average.append(np.mean(green))
    blues.append(np.mean(blue))
    reds.append(np.mean(red))
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    num_of_frames += 1

plt.subplot(3,1,1)
plt.plot(average,"green")
plt.subplot(3,1,2)
plt.plot(reds,"red")
plt.subplot(3,1,3)
plt.plot(blues,"blue")
plt.show()
