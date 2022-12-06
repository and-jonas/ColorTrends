import imageio
import utils
import glob
import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import random

# ======================================================================================================================
# Sample random patches from selected training images for manual annotation
# ======================================================================================================================

# from_dir = "Z:/Public/Jonas/003_ESWW/ColorTrends/train_img_selection/Selection_1200/"
from_dir = "Z:/Public/Jonas/003_ESWW/ColorTrends/train_img_selection/selection_soil_in_focus/"
# to_dir = "Z:/Public/Jonas/003_ESWW/ColorTrends/train_img_selection/patches_1200/"
to_dir = "Z:/Public/Jonas/003_ESWW/ColorTrends/train_img_selection/patches_soil_in_focus/"
images = glob.glob(f"{from_dir}*.JPG")

for image in images:
    base_name = os.path.basename(image)
    out_name = base_name.replace(".JPG", ".png")
    stem_name = out_name.replace(".png", "")
    print(base_name)
    img = imageio.imread(image)
    try:
        patch, coordinates = utils.random_patch(img, size=1200, frame=(750, 0, 750, 1500))
    except RuntimeError:
        continue
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"{to_dir}{out_name}", patch, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    coordinates = list(coordinates)
    df = pd.DataFrame([coordinates])
    df2 = df.set_axis(['x1', 'x2', 'y1', 'y2'], axis=1, inplace=False)
    df2.to_csv(f'{to_dir}/coordinates/{stem_name}.csv', index=False)

# ======================================================================================================================
# Make transforms of annotated patches
# 1) Patches of size 1750
# --> sample 4 patches of 1200x1200 px from these patches
# --> resize each to 1024x1024 px
# ======================================================================================================================

import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
from importlib import reload
reload(utils)

dir_img = "P:/Public/Jonas/003_ESWW/ColorTrends/train_img_selection/patches"
dir_mask = "P:/Public/Jonas/003_ESWW/ColorTrends/train_img_selection/annotations/annotations_20220218"
# dir_mask = "P:/Public/Jonas/003_ESWW/ColorTrends/train_img_selection/annotations/annotations_1200"
dir_to = "P:/Public/Jonas/003_ESWW/ColorTrends/train_img_selection/annotations/all_annotations"

# re-sample 1200x1200 patches from annotated patches
mask_paths = glob.glob(f'{dir_mask}/SegmentationClass/*.png')
for path in mask_paths:

    # set names
    base_name = os.path.basename(path)
    stem_name = base_name.replace(".png", "")
    img_name = base_name.replace(".png", ".JPG")
    img_out_name = img_name.replace(".JPG", ".png")

    # get binary mask
    mask = imageio.imread(path)
    mask_bin = np.zeros(mask.shape[:2]).astype("uint8")
    mask_bin = np.where(mask != (0, 0, 0), 255, 0)
    mask_bin = np.uint8(mask_bin[:, :, 0])

    # get image
    img = imageio.imread(f'{dir_img}/{img_name}')

    # re-sample patches
    img_patches, mask_patches = utils.sample_patches(img, mask_bin, size=1200)

    # save
    i = 1
    for img_patch, mask_patch in zip(img_patches, mask_patches):

        # make RGB
        img_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB)

        # # resize to 700x700 pixels
        # img_patch_resized = cv2.resize(img_patch, (700, 700), interpolation=cv2.INTER_LINEAR)
        # mask_patch_resized = cv2.resize(mask_patch, (700, 700), interpolation=cv2.INTER_NEAREST)

        # write
        cv2.imwrite(f"{dir_to}/full_size/Images/{stem_name}_{i}.png", img_patch, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        cv2.imwrite(f"{dir_to}/full_size/Masks/{stem_name}_{i}_mask.png", mask_patch, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        i += 1

# ======================================================================================================================
# Make transforms of annotated patches
# 2) Patches of size 1200
# --> resize to 700*700 px
# ======================================================================================================================

# dir_img = "P:/Public/Jonas/003_ESWW/ColorTrends/train_img_selection/patches"
# dir_img = "P:/Public/Jonas/003_ESWW/ColorTrends/train_img_selection/patches_1200"
dir_img = "Z:/Public/Jonas/Data/ESWW006/ImagesNadir/patches_annotation_spikes/STG_DIR/JPEGImages"
# dir_img = "P:/Public/Jonas/003_ESWW/ColorTrends/train_img_selection/patches_soil_in_focus"
# dir_mask = "P:/Public/Jonas/003_ESWW/ColorTrends/train_img_selection/annotations/annotations_1200"
# dir_mask = "P:/Public/Jonas/003_ESWW/ColorTrends/train_img_selection/annotations/annotations_20220406"
dir_mask = "Z:/Public/Jonas/Data/ESWW006/ImagesNadir/patches_annotation_spikes/STG_DIR/SegmentationClass"
# dir_mask = "P:/Public/Jonas/003_ESWW/ColorTrends/train_img_selection/annotations/annotations_20220923"
dir_to = "P:/Public/Jonas/003_ESWW/ColorTrends/train_img_selection/annotations/all_annotations"

# mask_paths = glob.glob(f'{dir_mask}/SegmentationClass/*.png')
mask_paths = glob.glob(f'{dir_mask}/*.png')
for path in mask_paths:

    # set names
    base_name = os.path.basename(path)
    stem_name = base_name.replace(".png", "")
    # img_name = base_name.replace(".png", ".JPG")
    img_name = base_name
    img_out_name = img_name.replace(".JPG", ".png")

    # get binary mask
    mask = imageio.imread(path)
    mask_bin = np.zeros(mask.shape[:2]).astype("uint8")
    mask_bin = np.where(mask != (0, 0, 0), 255, 0)
    mask_bin = np.uint8(mask_bin[:, :, 0])

    # get image
    img = imageio.imread(f'{dir_img}/{img_name}')

    # save
    # make RGB
    img_patch = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # # resize to 1024x1024 pixels
    # img_patch_resized = cv2.resize(img_patch, (700, 700), interpolation=cv2.INTER_LINEAR)
    # mask_patch_resized = cv2.resize(mask_bin, (700, 700), interpolation=cv2.INTER_NEAREST)

    # write
    # cv2.imwrite(f"{dir_to}/Images/{stem_name}.png", img_patch_resized, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    # cv2.imwrite(f"{dir_to}/Masks/{stem_name}_mask.png", mask_patch_resized, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite(f"{dir_to}/full_size/Images/{stem_name}.png", img_patch, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    cv2.imwrite(f"{dir_to}/full_size/Masks/{stem_name}_mask.png", mask_bin, [cv2.IMWRITE_PNG_COMPRESSION, 0])

# ======================================================================================================================

# some tests

import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt

mask_pred = imageio.imread("C:/Users/anjonas/PycharmProjects/DL/test_img.tiff")
img_test = imageio.imread("C:/Users/anjonas/PycharmProjects/DL/data/test/IMG_1467.png")
mask_ann = imageio.imread("C:/Users/anjonas/PycharmProjects/DL/data/test/IMG_1467_mask.png")

# Plot
fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
axs[0].imshow(img_test)
axs[0].set_title('img')
axs[1].imshow(mask_ann)
axs[1].set_title('mask_ann')
axs[2].imshow(mask_pred)
axs[2].set_title('mask_pred')
plt.show(block=True)

# ======================================================================================================================

# select some full images for inference

# selected images for training, validation and testing (do NOT use)
path1 = Path("Z:/Public/Jonas/003_ESWW/ColorTrends/train_img_selection/")
used_images = []
for p in path1.glob("Selection/*.JPG"):
    used_images.append(os.path.basename(p))

path2 = Path("Z:/Public/Jonas/003_ESWW/ColorTrends/train_img_selection/")
for p in path2.glob("Selection_1200/*.JPG"):
    used_images.append(os.path.basename(p))

# available images
base_path = "O:/Evaluation/FIP/2013/WW002/RGB/"
dirs = [base_path + d for d in os.listdir(base_path)[2:]]
files = [glob.glob(f'{d}/JPG/*.JPG') for d in dirs]
files = [item for sublist in files for item in sublist]

# identify already used ones and exclude
selected_files = []
for file in files:
    for image in used_images:
        if image in file:
            selected_files.append(file)
        else:
            pass
new_images = [file for file in files if file not in selected_files]

# randomly select 50 images
selected = random.choices(new_images, k=50)

# extract a central tile from images and save
to_dir = "Z:/Public/Jonas/003_ESWW/ColorTrends/train_img_selection/examples_inference"
for image in selected:
    base_name = os.path.basename(image)
    out_name = base_name.replace(".JPG", ".png")
    stem_name = out_name.replace(".png", "")
    print(base_name)
    img = imageio.imread(image)
    try:
        patch = utils.random_patch(img, size=1956, frame=(1614, 0, 1614, 1500), random_patch=False)
        patch = cv2.resize()
    except UnboundLocalError:
        continue
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(f"{to_dir}/{out_name}", patch, [cv2.IMWRITE_PNG_COMPRESSION, 0])

# ======================================================================================================================

# "pad" selected images for inference

selected = glob.glob("Z:/Public/Jonas/003_ESWW/ColorTrends/train_img_selection/examples_inference/orig/*.JPG")

# extract a central tile from images and save
to_dir = "Z:/Public/Jonas/003_ESWW/ColorTrends/train_img_selection/examples_inference/border"
for image in selected:
    base_name = os.path.basename(image)
    out_name = base_name.replace(".JPG", ".png")
    stem_name = out_name.replace(".png", "")
    print(base_name)
    img = imageio.imread(image)
    try:
        patch, coords = utils.random_patch(img, size=2178, frame=(1503, 0, 1503, 1278), random_patch=False)
    except UnboundLocalError:
        continue
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"{to_dir}/{out_name}", patch, [cv2.IMWRITE_PNG_COMPRESSION, 0])

# ======================================================================================================================



