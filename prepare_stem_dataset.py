
from pathlib import Path
import glob
import os
import imageio
import copy
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import utils

# ======================================================================================================================
# 1. Gather all annotations from ESWW006, ESWW007, and FPWW002
# ======================================================================================================================

all_annotated = glob.glob("P:/Public/Jonas/Data/ESWW006/ImagesNadir/patches_annotation_stems/All/*/SegmentationClass/*.png")
path_coordinates = "Z:/Public/Jonas/Data/ESWW006/ImagesNadir/Meta/patch_coordinates"
path_coordinates_patch = "Z:/Public/Jonas/Data/ESWW006/ImagesNadir/patches_annotation_spikes/coordinates"
path_original_images = "Z:/Public/Jonas/Data/ESWW006/ImagesNadir"
path_proc = "P:/Public/Jonas/Data/ESWW006/ImagesNadir/patches_annotation_stems/processed"

annotated_esww006 = [a for a in all_annotated if "ESWW0060" in a]

# ESWW006
for ann in annotated_esww006:
    # file names
    base_name = os.path.basename(ann)
    print(base_name)
    jpeg_name = base_name.replace(".png", ".JPG")
    stem_name = base_name.replace(".png", "")
    txt_name = f'{stem_name}.txt'
    mask_name = f'{stem_name}_mask.png'
    capture_date = stem_name.split("_")[0]
    capture_date = "_".join([capture_date[:4], capture_date[4:6], capture_date[6:8]])
    # get annotated sub-image
    image = f'{path_original_images}/{capture_date}/JPEG/{jpeg_name}'
    rc = tuple(pd.read_csv(f'{path_coordinates}/{txt_name}').iloc[0])
    img = imageio.imread(image)
    new_img = img[rc[2]:rc[3], rc[0]:rc[1]]
    rc = tuple(pd.read_csv(f'{path_coordinates_patch}/{txt_name}').iloc[0])
    # rc = tuple([275, 2975, 136, 2836])
    new_img = new_img[rc[2]:rc[3], rc[0]:rc[1]]
    imageio.imwrite(f'{path_proc}/{base_name}', new_img)
    # get mask
    mask = imageio.imread(ann)
    mask_scaled = cv2.resize(mask, (2700, 2700), interpolation=cv2.INTER_NEAREST)
    # mask_scaled = cv2.resize(mask, (2996, 2996), interpolation=cv2.INTER_NEAREST)  # for 4 im from 2022_05_30 (WTF)
    mask = np.where(mask_scaled == (250, 50, 83), 1, 0)
    if np.all(mask == 0):
        mask = np.where(mask_scaled == (255, 0, 124), 1, 0)
    mask = np.uint8(mask[:, :, 0])
    mask_checker = np.uint8(mask*255)
    imageio.imwrite(f'{path_proc}/{mask_name}', mask)
    # checker
    imageio.imwrite(f'{path_proc}/checker/{mask_name}', mask_checker)
    imageio.imwrite(f'{path_proc}/checker/{base_name}', new_img)

# FPWW002
annotated_fpww002 = [a for a in all_annotated if "fpww" in a]
for ann in annotated_fpww002:
    # file names
    base_name = os.path.basename(ann)
    print(base_name)
    jpeg_name = base_name.replace(".png", ".JPG")
    stem_name = base_name.replace(".png", "")
    mask_name = f'{stem_name}_mask.png'
    image_name = f'Z:/Public/Jonas/003_ESWW/ColorTrends/train_img_selection/patches_1200/{base_name}'
    # get annotated sub-image
    try:
        img = imageio.imread(image_name)
    except FileNotFoundError:
        image_name = ann.replace("SegmentationClass", "JPEGImages")
        img = imageio.imread(f'{image_name}')
    img = cv2.resize(img, (2700, 2700), interpolation=cv2.INTER_NEAREST)
    imageio.imwrite(f'{path_proc}/{base_name}', img)
    # get mask
    mask = imageio.imread(ann)
    mask_scaled = cv2.resize(mask, (2700, 2700), interpolation=cv2.INTER_NEAREST)
    # mask_scaled = cv2.resize(mask, (2996, 2996), interpolation=cv2.INTER_NEAREST)  # for 4 im from 2022_05_30 (WTF)
    mask = np.where(mask_scaled == (250, 50, 83), 1, 0)
    if np.all(mask == 0):
        mask = np.where(mask_scaled == (255, 0, 124), 1, 0)
    mask = np.uint8(mask[:, :, 0])
    mask_checker = np.uint8(mask*255)
    imageio.imwrite(f'{path_proc}/{mask_name}', mask)
    # checker
    imageio.imwrite(f'{path_proc}/checker/{mask_name}', mask_checker)
    imageio.imwrite(f'{path_proc}/checker/{base_name}', img)

# ESWW007
path_original_images1 = "Z:/Public/Jonas/Data/ESWW006/ImagesNadir/patches_annotation_stems/images_ESWW007_ESWW008/2700px"
annotated_esww007 = [a for a in all_annotated if "ESWW0070" in a]
for ann in annotated_esww007:
    # file names
    base_name = os.path.basename(ann)
    print(base_name)
    jpeg_name = base_name.replace(".png", ".JPG")
    stem_name = base_name.replace(".png", "")
    txt_name = f'{stem_name}.txt'
    mask_name = f'{stem_name}_mask.png'
    new_img = imageio.imread(f'{path_original_images1}/{base_name}')
    mask = imageio.imread(ann)
    mask_ = np.zeros((mask.shape[0], mask.shape[1]))
    idx1 = np.where(mask == (255, 0, 124))[:2]
    mask_[idx1] = 1
    mask_checker = np.uint8(mask_ * 255)
    imageio.imwrite(f'{path_proc}/{mask_name}', mask)
    # checker
    imageio.imwrite(f'{path_proc}/checker/{mask_name}', mask_checker)
    imageio.imwrite(f'{path_proc}/checker/{base_name}', new_img)


# ======================================================================================================================
# 2. Select angle images to annotated
# ======================================================================================================================

dir = "Z:/Public/Jonas/Data/ESWW006/ImagesNadir/patches_annotation_stems/images_ESWW007_ESWW008"
files = glob.glob(f"{dir}/Images/*.JPG")

for f in files:
    base_name = os.path.basename(f)
    out_name = base_name.replace(".JPG", ".png")
    stem_name = out_name.replace(".png", "")
    print(base_name)
    img = imageio.imread(f)
    patch, coords = utils.random_patch(img, size=608, frame=(0, 1250, 3000, 1250),
                                       random_patch=True, color_checker=False)
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"{dir}/608px/{out_name}", patch, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    df = pd.DataFrame([coords])
    df2 = df.set_axis(['x1', 'x2', 'y1', 'y2'], axis=1, inplace=False)
    df2.to_csv(f'{dir}/608px/coordinates/{stem_name}.txt', index=False)
    patch1 = img[coords[2]:coords[3], coords[0]:coords[1]]
    # Plot result
    # fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    # # Show RGB and segmentation mask
    # axs[0].imshow(patch)
    # axs[0].set_title('img')
    # axs[1].imshow(patch1)
    # axs[1].set_title('orig_mask')
    # plt.show(block=True)

# ======================================================================================================================
# 2. ESWW007 Nadir
# ======================================================================================================================

dir = "Z:/Public/Jonas/Data/ESWW006/ImagesNadir/patches_annotation_stems/images_ESWW007_ESWW008"
files = glob.glob(f"{dir}/Images/Nadir/*.JPG")

for f in files:
    base_name = os.path.basename(f)
    out_name = base_name.replace(".JPG", ".png")
    stem_name = out_name.replace(".png", "")
    print(base_name)
    img = imageio.imread(f)
    patch, coords = utils.random_patch(img, size=2700, frame=(0, 1250, 3000, 1250),
                                       random_patch=True, color_checker=False)
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    cv2.imwrite(f"{dir}/2700px/{out_name}", patch, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    df = pd.DataFrame([coords])
    df2 = df.set_axis(['x1', 'x2', 'y1', 'y2'], axis=1, inplace=False)
    df2.to_csv(f'{dir}/2700px/coordinates/{stem_name}.txt', index=False)
    patch1 = img[coords[2]:coords[3], coords[0]:coords[1]]
    # Plot result
    # fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    # # Show RGB and segmentation mask
    # axs[0].imshow(patch)
    # axs[0].set_title('img')
    # axs[1].imshow(patch1)
    # axs[1].set_title('orig_mask')
    # plt.show(block=True)