import os
import glob
import random
import imageio
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import utils
from importlib import reload
reload(utils)
import numpy as np
import cv2

path = "Z:/Public/Jonas/003_ESWW/ColorTrends/TrainSoil"

os.chdir(path)

filenames = glob.glob('*.png')

image_list = []
for filename in filenames:
    img = imageio.imread(filename)[:, :, :3]
    image_list.append(img)

random_soil = random.choice(image_list)
plt.imshow(random_soil)

hists = utils.get_hist(random_soil)

distances = []
for img in image_list:
    hists_img = utils.get_hist(img)
    dist = sum([abs(x) for x in (hists_img[0] - hists[0])])
    distances.append(dist)

# remove the zero value
distances = [i for i in distances if i != 0.0]
# find smallest difference
min_value = np.min(distances)
min_index = distances.index(min_value)

similar_soil = image_list[18]
plt.imshow(similar_soil)
plt.imshow(random_soil)


# ======

testimg = imageio.imread("Z:/Public/Jonas/003_ESWW/ColorTrends/TrainSoil/Testimg/FPWW0320061_RGB1_20210721_142245.jpg")

text = utils.quilt(testimg, 50, (10, 10), "best", True)

plt.imshow(text)

