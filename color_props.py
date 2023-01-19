
import glob
import re
import imageio
import pandas as pd
import numpy as np
import os
from pathlib import Path

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import utils

# ======================================================================================================================
# TEST
# ======================================================================================================================

# list files to process
data_from = "Z:/Public/Jonas/003_ESWW/ColorTrends/train_img_selection/annotations/all_annotations/full_size"
images = glob.glob(f'{data_from}/Images/*.png')
masks = [re.sub(".png", "_mask.png", image) for image in images]
masks = [re.sub("Images", "Masks", mask) for mask in masks]

# extract stats for each color feature
D = []
for i in range(len(images)):
    print(i)
    base_name = os.path.basename(images[i]).replace(".png", "")
    image = imageio.imread(images[i])
    mask = imageio.imread(masks[i])
    desc, desc_names = utils.color_index_transformation(image)
    df = pd.DataFrame()
    for d, d_n in zip(desc, desc_names):
        stats, stat_names = utils.index_distribution(image=d, image_name=d_n, mask=mask)
        df[stat_names] = [stats]
    df.insert(loc=0, column='image_id', value=images[i])
    df.insert(loc=1, column='image_name', value=base_name)
    D.append(df)
STATS = pd.concat(D, axis=0)
STATS.to_csv("Z:/Public/Jonas/003_ESWW/Results/PostSegmentation/col_feat_dist.csv",
             index=False)

# ======================================================================================================================
# REAL
# ======================================================================================================================

coordinates_path = "Z:/Public/Jonas/Data/ESWW006/ImagesNadir/Meta/patch_coordinates"
original_image = Path("Z:/Public/Jonas/Data/ESWW006/ImagesNadir/2022_06_23/JPEG/20220623_Cam_ESWW0060058_Cnp_1.JPG")
image_name = os.path.basename(original_image)
image_base_name = image_name.replace(".JPG", "")

# get image
image = imageio.imread(original_image)
mask = imageio.imread("Z:/Public/Jonas/Data/ESWW006/ImagesNadir/Output/SegEar/Mask/20220623_Cam_ESWW0060058_Cnp_1.png")

# sample patch from image
c = pd.read_table(f'{coordinates_path}/{image_base_name}.txt', sep=",").iloc[0, :].tolist()
patch = image[c[2]:c[3], c[0]:c[1]]

desc, desc_names = utils.color_index_transformation(patch)
df = pd.DataFrame()
for d, d_n in zip(desc, desc_names):
    stats, stat_names = utils.index_distribution(image=patch, image_name=d_n, mask=mask)
    df[stat_names] = [stats]
df.insert(loc=0, column='image_id', value=images[i])
df.insert(loc=1, column='image_name', value=base_name)
D.append(df)




mask_3d = np.stack([mask/255, mask/255, mask/255], axis=2)

roi = patch * np.uint8(mask_3d)


plt.imshow(roi)


fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
axs[0].imshow(patch)
axs[0].set_title('img')
axs[1].imshow(mask)
axs[1].set_title('orig_mask')
plt.show(block=True)



# fig, axs = plt.subplots(1, 8, sharex=True, sharey=True)
# for i in range(len(desc)):
#     axs[i].imshow(D[i])
#     axs[i].set_title(desc_names[i])
# axs[7].imshow(image)
# axs[7].set_title('image')
# plt.show(block=True)
# fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
# axs[0].imshow(D[2])
# axs[0].set_title('img')
# axs[1].imshow(image)
# axs[1].set_title('orig_mask')
# plt.show(block=True)
