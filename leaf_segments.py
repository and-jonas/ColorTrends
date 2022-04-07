import imageio
import cv2
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np

image = "O:/Evaluation/FIP/2013/WW002/RGB/2013-07-16_WW002_145-216/JPG/IMG_1082.JPG"
img = imageio.imread(image)

# get yellow pixels
lower_yellow = np.array([170, 120, 0])
upper_yellow = np.array([255, 255, 120])
mask_yellow = cv2.inRange(img, lower_yellow, upper_yellow)

img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# get yellow pixels
lower_gray = 150
upper_gray = 255
mask_gray = cv2.inRange(img_gray, lower_gray, upper_gray)

fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
axs[0].imshow(img_gray)
axs[0].set_title('original')
axs[1].imshow(mask_gray)
axs[1].set_title('soil patch')
axs[2].imshow(img)
axs[2].set_title('soil patch')
plt.show(block=True)



fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
axs[0, 0].imshow(img)
axs[0, 0].set_title('original')
axs[0, 1].imshow(mask_yellow)
axs[0, 1].set_title('soil patch')
axs[1, 0].imshow(img_gray)
axs[1, 0].set_title('original')
axs[1, 1].imshow(mask_gray)
axs[1, 1].set_title('soil patch')
plt.show(block=True)