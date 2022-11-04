import os
import matplotlib as mpl
import pandas as pd

mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import imageio
import numpy as np
import cv2
import utils
import PIL
import copy
from importlib import reload

reload(utils)
import glob
import scipy
from pathlib import Path
import shutil
import random

# ======================================================================================================================

# # directories
# path = "Z:/Public/Jonas/Data/ESWW006/Images_trainset"
# out_dir = f"{path}/Output/synthetic_images"
# plant_dir = f"{path}/RectPatches"
# mask_dir = f"{path}/Output/Mask"

# # directories
# path = "Z:/Public/Jonas/Data/ESWW006/Images_trainset/subset_1"
# out_dir = f"{path}/Output/synthetic_images"
# plant_dir = f"{path}/patches"
# mask_dir = f"{path}/masks"

# indicate batches and corresponding soil batches
batch_nr = [1, 2, 3, 4]
soil_type = ["dif", "dif", "dif", "dir_dif"]

# iterate over all batches
for b, s in zip(batch_nr, soil_type):

    # directories
    path = "Z:/Public/Jonas/Data/ESWW006/Images_trainset/"
    out_dir = f"{path}Output/synthetic_images/{b}"
    plant_dir = f"{path}PatchSets/{b}/patches"
    mask_dir = f"{path}Output/annotations_manual/{b}/SegmentationClass"
    soil_paths = glob.glob(
        f"Z:/Public/Jonas/Data/ESWW006/Images_trainset/Soil/{s}/*.JPG")

    # list plant images and masks
    plants = glob.glob(f'{plant_dir}/*.JPG')
    masks = glob.glob(f'{mask_dir}/*.png')

    # iterate over all plants
    for p, m in zip(plants, masks):

        stem_name = os.path.basename(p).replace(".JPG", "")
        Plot_ID, date = stem_name.split("_")

        img = imageio.imread(p)
        mask = imageio.imread(m)
        # check if mask is binary; binarize if needed
        if not len(mask.shape) == 2:
            mask = utils.binarize_mask(mask)

        # erode original mask to get rid of the blueish pixels along plant edges
        mask_erode = cv2.erode(mask, np.ones((2, 2), np.uint8))

        # dilate original mask and invert to obtain a soil mask with a "safety margin" along plant edges
        mask_dilate = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        mask_dilate_inv = cv2.bitwise_not(mask_dilate)

        # get connected components
        _, labels_fg, _, _ = cv2.connectedComponentsWithStats(mask_dilate)
        _, labels_bg, _, _ = cv2.connectedComponentsWithStats(mask_dilate_inv)

        # isolate soil patches in original image, mask plant
        img_ = np.zeros_like(img)
        idx = np.where(labels_fg == 0)
        img_[idx] = img[idx]

        # convert the original image with plant masked to gray scale
        soil_gray = cv2.cvtColor(img_, cv2.COLOR_RGB2GRAY)
        soil_gray = soil_gray / soil_gray.max()

        # perform in-filling along the leaf-background boundaries to fill the "safety margin"
        mask_eroded = cv2.erode(mask_dilate, np.ones((9, 9), np.uint8))
        mask_inpaint = ((mask_dilate - mask_eroded) / 255).astype("uint8")
        sg = (soil_gray * 255).astype("uint8")
        final = cv2.inpaint(sg, mask_inpaint, 1, cv2.INPAINT_TELEA) / 255

        # ==============================================================================================================

        # The above "region growing" does not work for small holes
        # repair by comparing these holes with the holes in the eroded vegetation mask

        # binarize and invert
        bin = np.uint8(np.where(final != 0, 255, 0))
        bin = np.bitwise_not(bin)

        mask_inverted = np.bitwise_not(mask_erode)
        mask_sizefiltered = utils.filter_objects_size(mask_inverted, size_th=500, dir="greater")

        # holes present in the eroded mask, but not in the dilated one used before
        diff_mask = np.bitwise_and(bin, mask_sizefiltered)

        # paste the original content of the holes onto empty image
        _, l, _, _ = cv2.connectedComponentsWithStats(mask_sizefiltered)
        img2_ = np.zeros_like(img)
        idx = np.where(l != 0)
        img2_[idx] = img[idx]

        # convert to gray scale as done for the rest of the soil background
        # and combine with the previous soil intensity image
        soil_gray2 = cv2.cvtColor(img2_, cv2.COLOR_RGB2GRAY)
        soil_gray2 = soil_gray2 / soil_gray2.max()
        soil_gray2 = np.where(diff_mask == 255, soil_gray2, final)

        # final post-processing
        final_soil_gray = scipy.ndimage.percentile_filter(np.uint8(soil_gray2 * 255),
                                                          percentile=60, size=7, mode='reflect')
        final_soil_gray = final_soil_gray / 255

        # ==============================================================================================================

        # iterate over all soils
        for soil_path in soil_paths:

            soil_name = os.path.basename(soil_path)
            soil_image = imageio.imread(soil_path)
            soil, coordinates = utils.get_soil_patch(image=soil_image, size=(2400, 2400))

            if soil is None:
                continue

            r, g, b = cv2.split(soil[:2400, :2400, :3])
            r_ = r * final_soil_gray
            g_ = g * final_soil_gray
            b_ = b * final_soil_gray
            img_ = cv2.merge((r_, g_, b_))
            img_ = np.uint8(img_)

            # ==============================================================================================================

            # overlay eroded plant mask to the created soil background
            transparency = np.ones_like(img_[:, :, 0]) * 255
            transparency[np.where(mask_erode == 255)] = 0
            img_final = np.dstack([img_, transparency])
            final = PIL.Image.fromarray(np.uint8(img_final))
            final = final.convert("RGBA")
            img2 = PIL.Image.fromarray(np.uint8(img))
            img2.paste(final, (0, 0), final)
            final_patch = np.asarray(img2)

            # ==============================================================================================================

            # remove last remaining holes between soil and eroded plant mask
            # get remaining holes and dilate them
            mmm = np.zeros_like(final_patch)
            idx = np.where(np.all(final_patch == 0, axis=-1))
            mmm[idx] = [255, 255, 255]
            mmm = mmm[:, :, 0]
            kernel = np.ones((4, 4), np.uint8)
            mmm_d = cv2.dilate(mmm, kernel)
            filter_mask = np.dstack([mmm_d, mmm_d, mmm_d])

            # blur the final image and multiply with the hole mask
            bblurred = cv2.blur(final_patch, (10, 10)) * filter_mask
            hole_filler = bblurred * np.dstack([mmm, mmm, mmm])

            # fill the holes
            filled_holes = final_patch + hole_filler

            # ==============================================================================================================

            # blur edges
            edge_mask_thin = np.zeros_like(mask_erode)
            edge_mask = np.zeros_like(mask_erode)
            contours, hier = cv2.findContours(mask_erode, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            for c in contours:
                cv2.drawContours(edge_mask, c, -1, color=1, thickness=6)
                cv2.drawContours(edge_mask_thin, c, -1, color=1, thickness=3)

            # blur the final image and multiply with the hole mask
            blurred_edges = cv2.blur(filled_holes, (2, 2)) * np.dstack([edge_mask, edge_mask, edge_mask])

            # replace "original" edges with blurred edges
            idx = np.where(edge_mask_thin == 1)
            filled_holes[idx] = blurred_edges[idx]

            out_name = soil_name.replace(".JPG", ".png")
            img_name = f'{stem_name}_{out_name}'
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            synth_img_name = Path(out_dir) / img_name
            synth_image = filled_holes
            imageio.imwrite(synth_img_name, synth_image)

# ======================================================================================================================

# divide synthetic images into 4 patches of 1200px x 1200px each

import utils
import random

directory = "Z:/Public/Jonas/Data/ESWW006/images_trainset/Output/synthetic_images/5"
images = glob.glob(f'{directory}/*.png')

out_dir = "Z:/Public/Jonas/Data/ESWW006/images_trainset/Output/CGAN_input/composite2real_int_dir"

# for cyclegan

trainB = random.sample(images, k=round(len(images) * 0.8))
testB = [item for item in images if item not in trainB]

for im in trainB:
    img = imageio.imread(im)
    img_name = os.path.basename(im)
    tiles = utils.image_tiler(img, stride=1200)
    for i, tile in enumerate(tiles):
        out_name = img_name.replace(".png", f"_{i + 1}.jpg")
        imageio.imwrite(f"{out_dir}/trainB/{out_name}", tile)

# ======================================================================================================================
# RUN THE CYCLE GAN
# ======================================================================================================================

# ======================================================================================================================
# PREPARE DATA FOR SEGMENTATION TRAINING
# ======================================================================================================================

import random

# process masks first
# dir_masks = "Z:/Public/Jonas/Data/ESWW006/images_trainset/Output/annotations_manual/all"
dir_masks = "Z:/Public/Jonas/Data/ESWW006/images_trainset/Output/annotations_manual/5/SegmentationClass"
out_dir_masks = "Z:/Public/Jonas/Data/ESWW006/images_trainset/Output/annotations_manual/5_patches"

# tile masks
# masks = glob.glob(f'{dir_masks}/*_mask.png')
masks = glob.glob(f'{dir_masks}/*.png')
for m in masks:
    mask = imageio.imread(m)
    mask_name = os.path.basename(m)
    mask_bin = utils.binarize_mask(mask)
    mask_tiles = utils.image_tiler(mask_bin, stride=1200)
    for j in range(len(mask_tiles)):
        # out_name = mask_name.replace("_mask.png", f"_{j+1}_mask.png")
        out_name = mask_name.replace(".png", f"_{j + 1}_mask.png")
        imageio.imwrite(f"{out_dir_masks}/{out_name}", mask_tiles[j])

# create train, test, validation from cyclegan output

files = glob.glob(
    "C:/Users/anjonas/PycharmProjects/pytorch-CycleGAN-and-pix2pix/results/wheat_cyclegan/test_latest/images/*_fake.png")


def get_plot_id(file_names):
    plot_ids = []
    for name in file_names:
        n = os.path.basename(name)
        plot_ids.append(n.split("_")[0])
    return plot_ids


plots = get_plot_id(files)
plots = np.unique(plots)

n_plots = len(np.unique(plots))
n_train = int(np.ceil(0.75 * n_plots))
n_test = int((n_plots - n_train) / 2)
n_val = int(n_test)

random.seed(10)
train_plots = random.sample(list(plots), k=n_train)
not_train = set(plots) - set(train_plots)
random.seed(10)
test_plots = random.sample(list(not_train), k=n_test)
val_plots = list(set(not_train) - set(test_plots))

train_list = []
for p in train_plots:
    lst = [s for s in files if p in s]
    train_list.extend(lst)

test_list = []
for p in test_plots:
    lst = [s for s in files if p in s]
    test_list.extend(lst)

val_list = []
for p in val_plots:
    lst = [s for s in files if p in s]
    val_list.extend(lst)

lst = train_list + test_list + val_list
lst_key = ["train"] * len(train_list) + ["test"] * len(test_list) + ["validation"] * len(val_list)

out_dir = "C:/Users/anjonas/PycharmProjects/SegVeg/data/3"

# move images
for ele, key in zip(lst, lst_key):
    name = os.path.basename(ele)
    img_id = name.split("_")[:2]
    img_id = "_".join(img_id)
    patch_id = name.split("_")[len(name.split("_")) - 2]
    destination = f'{out_dir}/{key}/{name}'
    img = imageio.imread(ele)
    img_resized = cv2.resize(img, (700, 700), interpolation=cv2.INTER_LINEAR)
    imageio.imwrite(destination, img_resized)
    mask = imageio.imread(f'{out_dir_masks}/{img_id}_{patch_id}_mask.png')
    mask_resized = cv2.resize(mask, (700, 700), interpolation=cv2.INTER_NEAREST)
    mask_name = name.replace(".png", "_mask.png")
    destination_mask = f'{out_dir}/{key}/{mask_name}'
    imageio.imwrite(destination_mask, mask_resized)

# ======================================================================================================================
# COMBINE MULTIPLE TYPES FOR TRAINING
# ======================================================================================================================

# list all transformed images
files = glob.glob("D:/SegVeg/[0-9]/*/*_fake.png")

plots = get_plot_id(files)
plots = np.unique(plots)

n_plots = len(np.unique(plots))
n_train = int(np.ceil(0.8 * n_plots))
n_val = int((n_plots - n_train))

random.seed(10)
train_plots = random.sample(list(plots), k=n_train)
val_plots = set(plots) - set(train_plots)

train_list = []
for p in train_plots:
    lst = [s for s in files if p in s]
    train_list.extend(lst)

val_list = []
for p in val_plots:
    lst = [s for s in files if p in s]
    val_list.extend(lst)

lst = train_list + val_list
lst_key = ["train"] * len(train_list) + ["validation"] * len(val_list)

out_dir = "C:/Users/anjonas/PycharmProjects/SegVeg/data/combined"

# move images
for ele, key in zip(lst, lst_key):
    name = os.path.basename(ele)
    img_id = name.split("_")[:2]
    img_id = "_".join(img_id)
    patch_id = name.split("_")[len(name.split("_")) - 2]
    destination = f'{out_dir}/{key}/{name}'
    shutil.copy(ele, destination)
    mask_source = ele.replace(".png", "_mask.png")
    mask_destination = destination.replace(".png", "_mask.png")
    shutil.copy(mask_source, mask_destination)

#
# # for direct evaluation
#
# directory = "Z:/Public/Jonas/Data/ESWW006/images_trainset/Output/synthetic_images"
# dir_masks = "Z:/Public/Jonas/Data/ESWW006/images_trainset/Output/annotations_manual/all"
# images = glob.glob(f'{directory}/*.png')
#
# out_dir = "C:/Users/anjonas/PycharmProjects/SegVeg/data"
#
# image_id = []
# for im in images:
#     base_name = os.path.basename(im).split("_")
#     base_name = "_".join(base_name[:2])
#     image_id.append(base_name)
# image_id = np.unique(image_id).tolist()
#
# train = random.sample(image_id, k=61)
# test_val = [item for item in image_id if item not in train]
# test = random.sample(test_val, k=10)
# val = [item for item in test_val if item not in test]
#
# for im in val:
#     all_ims = glob.glob(f'{directory}/{im}*.png')
#     for i in all_ims:
#         img = imageio.imread(i)
#         img_name = os.path.basename(i)
#         mask = imageio.imread(f'{dir_masks}/{im}_mask.png')
#         # binarize mask
#         mask = utils.binarize_mask(mask)
#         tiles = utils.image_tiler(img, stride=1200)
#         mask_tiles = utils.image_tiler(mask, stride=1200)
#         for j in range(len(tiles)):
#             # resize to 700x700 pixels
#             img_patch_resized = cv2.resize(tiles[j], (700, 700), interpolation=cv2.INTER_LINEAR)
#             mask_patch_resized = cv2.resize(mask_tiles[j], (700, 700), interpolation=cv2.INTER_NEAREST)
#             out_name = img_name.replace(".png", f"_{j+1}.png")
#             imageio.imwrite(f"{out_dir}/val/{out_name}", img_patch_resized)
#             out_name = img_name.replace(".png", f"_{j+1}_mask.png")
#             imageio.imwrite(f"{out_dir}/val/{out_name}", mask_patch_resized)
#
#
# # real images
# dir = "Z:/Public/Jonas/Data/ESWW006/images_trainset/Output/annotations_manual/real_images"
# out_dir = "C:/Users/anjonas/PycharmProjects/SegVeg/data"
#
# all_patches = glob.glob(f'{dir}/JPEGImages/*.png')
#
# for p in all_patches:
#     img = imageio.imread(p)
#     img_name = os.path.basename(p)
#     mask_name = img_name.replace(".png", "_mask.png")
#     mask = imageio.imread(f'{dir}/SegmentationClass/{img_name}')
#     # binarize mask
#     mask = utils.binarize_mask(mask)
#     imageio.imwrite(f"{out_dir}/val_ext/{img_name}", img)
#     imageio.imwrite(f"{out_dir}/val_ext/{mask_name}", mask)
#
#
#
# fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
# axs[0].imshow(tiles[0])
# axs[0].set_title('img')
# axs[1].imshow(mask_tiles[0])
# axs[1].set_title('orig_mask')
# plt.show(block=True)
#
#
# ======================================================================================================================
# make same name files for target and image and move to separate folders
# ======================================================================================================================

images = glob.glob("C:/Users/anjonas/PycharmProjects/SegVeg/data/combined/*/*_fake.png")
masks = glob.glob("C:/Users/anjonas/PycharmProjects/SegVeg/data/combined/*/*_mask.png")

to_dir = "Z:/Public/Jonas/Data/ESWW006/Images_trainset/Output/cGAN_output"

for img in images:
    from_dir = img
    base_name = os.path.basename(from_dir)
    target = to_dir + "/images/" + base_name
    shutil.copy(from_dir, target)

for msk in masks:
    from_dir = msk
    base_name = os.path.basename(from_dir)
    target = to_dir + "/masks/" + base_name
    shutil.copy(from_dir, target)

# ======================================================================================================================
# make same name files for target and image and move to separate folders
# ======================================================================================================================

## TRAINING

train_images = glob.glob("C:/Users/anjonas/PycharmProjects/SegVeg/data/combined/train/*_fake.png")
train_masks = glob.glob("C:/Users/anjonas/PycharmProjects/SegVeg/data/combined/train/*_mask.png")

to_dir = "Z:/Public/Jonas/Data/ESWW006/Images_trainset/Output/cGAN_output/train"

# random background
base_names = [os.path.basename(x) for x in train_images]

plots = []
dates = []
soils = []
patches = []
for b in base_names:
    eles = b.split("_")
    plots.append(eles[0])
    dates.append(eles[1])
    patches.append(eles[len(eles) - 2])
    soil = eles[2:len(eles) - 2]
    if len(soil) > 1:
        soil = "_".join(soil)
    else:
        soil = soil[0]
    soils.append(soil)

df = pd.DataFrame(list(zip(train_images, plots, dates, soils, patches)),
                  columns=['fullname', 'plot', 'date', 'soil', 'patch'])

# per plot and measurement event ("date", pseudo-date), select one scenario
# (i.e. one random soil background, all 4 patches)
random.seed(10)
df2 = df.groupby(['plot', 'date', 'patch']).apply(lambda x: x.sample(1, random_state=10)).reset_index(drop=True)
used = df2['fullname'].tolist()

for img in used:
    print(img)
    from_dir = img
    base_name = os.path.basename(from_dir)
    target = to_dir + "/images/" + base_name
    shutil.copy(from_dir, target)

for msk in used:
    from_dir = msk.replace(".png", "_mask.png")
    base_name = os.path.basename(from_dir)
    base_name_corr = base_name.replace("_mask.png", ".png")
    target = to_dir + "/masks/" + base_name_corr
    mask = imageio.imread(from_dir)
    mask_bin = (mask/255).astype("uint8")
    imageio.imwrite(target, mask_bin)

# ======================================================================================================================

## VALIDATION

val_images = glob.glob("C:/Users/anjonas/PycharmProjects/SegVeg/data/combined/validation/*_fake.png")
val_masks = glob.glob("C:/Users/anjonas/PycharmProjects/SegVeg/data/combined/validation/*_mask.png")

to_dir = "Z:/Public/Jonas/Data/ESWW006/Images_trainset/Output/cGAN_output/validation"

# random background
base_names = [os.path.basename(x) for x in val_images]

plots = []
dates = []
soils = []
patches = []
for b in base_names:
    eles = b.split("_")
    plots.append(eles[0])
    dates.append(eles[1])
    patches.append(eles[len(eles) - 2])
    soil = eles[2:len(eles) - 2]
    if len(soil) > 1:
        soil = "_".join(soil)
    else:
        soil = soil[0]
    soils.append(soil)

df = pd.DataFrame(list(zip(val_images, plots, dates, soils, patches)),
                  columns=['fullname', 'plot', 'date', 'soil', 'patch'])

# per plot and measurement event ("date", pseudo-date), select one scenario
# (i.e. one random soil background, all 4 patches)
random.seed(10)
df2 = df.groupby(['plot', 'date', 'patch']).apply(lambda x: x.sample(1, random_state=10)).reset_index(drop=True)
used = df2['fullname'].tolist()

for img in used:
    print(img)
    from_dir = img
    base_name = os.path.basename(from_dir)
    target = to_dir + "/images/" + base_name
    shutil.copy(from_dir, target)

for msk in used:
    from_dir = msk.replace(".png", "_mask.png")
    base_name = os.path.basename(from_dir)
    base_name_corr = base_name.replace("_mask.png", ".png")
    target = to_dir + "/masks/" + base_name_corr
    mask = imageio.imread(from_dir)
    mask_bin = (mask/255).astype("uint8")
    imageio.imwrite(target, mask_bin)

# ======================================================================================================================

# inference

path = "Z:/Public/Jonas/Data/ESWW006/ImagesNadir/patches_annotation/_TEST_"
out_path = "Z:/Public/Jonas/Data/ESWW006/ImagesNadir/patches_annotation/_TEST_/ImageResize"

files = glob.glob(f'{path}/*.png')

for file in files:
    img = imageio.imread(file)
    img_resized = cv2.resize(img, (700, 700), interpolation=cv2.INTER_LINEAR)
    base_name = os.path.basename(file)
    out_name = base_name.replace(".jpg", ".png")
    imageio.imwrite(f"{out_path}/{out_name}", img_resized)

# ======================================================================================================================

IMG = imageio.imread("C:/Users/anjonas/PycharmProjects/segveg2/data/CameraSeg/F61-1.png")
img = imageio.imread("C:/Users/anjonas/PycharmProjects/SegVeg/data/combined/validation/ESWW2015_202207063_BF0A0036_1_fake_mask.png")
