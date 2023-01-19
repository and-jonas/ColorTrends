import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import os
import pandas as pd
import imageio
import numpy as np
import cv2
import utils
import PIL
import glob
import scipy
from pathlib import Path
import shutil
import random
import re

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
batch_nr = [1, 2, 3, 4,
            5, 6, 7, 8,
            10, 11]
soil_type = ["dif", "dif", "dif", "dir_dif",
             "dir_dif", "dif", "dir_dif", "dir_dif",
             "dir_dif", "dir_dif"]
# batch_nr = [8, 10]
# soil_type = ["dir_dif", "dir_dif"]
save_masks = True
n_soils_per_image = 10

# iterate over all batches
for b, st in zip(batch_nr, soil_type):

    # directories
    path = "Z:/Public/Jonas/Data/ESWW006/Images_trainset/"
    out_dir = f"{path}Output/synthetic_images/{b}"
    plant_dir = f"{path}PatchSets/{b}/patches"
    mask_dir = f"{path}Output/annotations_manual/{b}/SegmentationClass"
    soil_paths = glob.glob(
        f"Z:/Public/Jonas/Data/ESWW006/Images_trainset/Soil/{st}/*.JPG")
    masks_out_dir = f"{path}Output/annotations_manual/masks"
    masks_out_dir_8bit = f"{path}Output/annotations_manual/masks/8bit"
    edited_dir = f"{path}PatchSets/{b}/edited.txt"
    processed_dir = f"{path}Output/synthetic_images/{b}/"
    # create directories
    Path(masks_out_dir).mkdir(exist_ok=True, parents=True)
    Path(masks_out_dir_8bit).mkdir(exist_ok=True, parents=True)

    # list plant images and masks
    plants = glob.glob(f'{plant_dir}/*.JPG')
    masks = glob.glob(f'{mask_dir}/*.png')

    # only process edited ones
    if Path(edited_dir).exists():
        edited_list = pd.read_csv(edited_dir, header=None).iloc[:, 0].tolist()
        plants = [ele for ele in plants if os.path.basename(ele).replace(".JPG", "") in edited_list]
        masks = [ele for ele in masks if os.path.basename(ele).replace(".png", "") in edited_list]

    # only process new ones
    processed = glob.glob(f'{processed_dir}/*.png')
    processed_list = [utils.get_plot(ele) for ele in processed]
    plants = [ele for ele in plants if os.path.basename(ele).replace(".JPG", "") not in processed_list]
    masks = [ele for ele in masks if os.path.basename(ele).replace(".png", "") not in processed_list]

    # iterate over all plants
    for p, m in zip(plants, masks):

        print(p)

        counter = None

        base_name = os.path.basename(p)
        stem_name = base_name.replace(".JPG", "")
        Plot_ID, date = stem_name.split("_")

        img = imageio.imread(p)
        mask = imageio.imread(m)

        # check if mask is binary; binarize if needed
        if not len(mask.shape) == 2:
            mask = utils.binarize_mask(mask)
        # save
        imageio.imwrite(f"{masks_out_dir}/{stem_name}.png", np.uint8(mask/255))
        imageio.imwrite(f"{masks_out_dir_8bit}/{stem_name}_mask.png", mask)
        imageio.imwrite(f"{masks_out_dir_8bit}/{base_name}", img)

        # list already generated composites
        if Path(out_dir).exists():
            existing = glob.glob(f'{out_dir}/{Plot_ID}_{date}*.png')
            existing = [ele for ele in existing if utils.get_plot(ele) == stem_name]

            # make 10 composites
            if len(existing) >= n_soils_per_image:
                continue
            else:
                counter = len(existing)

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
        # TODO is there a faster way to do this?
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

        # increase the contrast between shaded and sunlight portions of the background
        # TODO this results in a step, must be corrected!!!
        adjusted = np.where(final_soil_gray <= 0.4, final_soil_gray, 1.5 * final_soil_gray - 0.1)
        adjusted = np.where(adjusted < 0, 0, adjusted)

        # fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
        # axs[0].imshow(final_soil_gray)
        # axs[0].set_title('original')
        # axs[1].imshow(adjusted)
        # axs[1].set_title('transformed')
        # axs[2].imshow(img)
        # axs[2].set_title('image')
        # plt.show(block=True)

        # ==============================================================================================================

        # randomly select 15 unused soils per image
        used = [utils.get_soil_id(x) for x in existing]
        soil_paths_unused = [x for x in soil_paths if os.path.basename(x).replace(".JPG", "") not in used]
        soils = random.sample(soil_paths_unused, k=int(np.ceil(n_soils_per_image + 0.5*n_soils_per_image)))

        # iterate over all soils
        counter = counter if counter is not None else 1
        for s in soils:

            if counter == n_soils_per_image:
                break

            soil_name = os.path.basename(s)
            soil_image = imageio.imread(s)
            soil, coordinates = utils.get_soil_patch(image=soil_image, size=(2400, 2400))

            if soil is None:
                continue

            img_ = utils.apply_intensity_map(soil, adjusted)

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

            counter += 1

# ======================================================================================================================
# split into training and test set for cGAN
# ======================================================================================================================

import utils
import random

directory = "Z:/Public/Jonas/Data/ESWW006/images_trainset/Output/synthetic_images/[0-9]*"
images = glob.glob(f'{directory}/*.png')

out_dir = "Z:/Public/Jonas/Data/ESWW006/images_trainset/Output/CGAN_input/composite2real_int"
Path(out_dir).mkdir(parents=True, exist_ok=True)

df = pd.DataFrame({'name': images})
df['id'] = utils.get_identifier(images)
df2 = df.groupby(['id']).apply(lambda x: x.sample(10, random_state=10)).reset_index(drop=True)
used = df2['name'].tolist()

# for cycle-gan
random.seed(10)
trainB = random.sample(used, k=round(len(used) * 0.8))
testB = [item for item in used if item not in trainB]

tile = False
for im in trainB:
    img_name = os.path.basename(im)
    img = imageio.imread(im)
    if tile:
        tiles = utils.image_tiler(img, stride=1200)
        for i, tile in enumerate(tiles):
            out_name = img_name.replace(".png", f"_{i + 1}.jpg")
            imageio.imwrite(f"{out_dir}/trainB/{out_name}", tile)
    else:
        out_name = img_name.replace(".png", ".jpg")
        imageio.imwrite(f"{out_dir}/trainB/{out_name}", img)

# real images for cycle gan training:
# image_selection_real.py

# move images needing domain transfer to folder
existing_folder = "Z:/Public/Jonas/Data/ESWW006/images_trainset/Output/CGAN_output/v1/*"
existing_images = glob.glob(f'{existing_folder}/*_fake.png')
existing_images = np.unique(utils.get_identifier(existing_images))

raw_folder = "Z:/Public/Jonas/Data/ESWW006/images_trainset/Output/synthetic_images/[0-9]*"
raw_composites = glob.glob(f'{raw_folder}/*.png')
rc = np.unique(utils.get_identifier(raw_composites))

to_process = [ele for ele in rc if ele not in existing_images]
to_process = [ele for ele in raw_composites if any(x in ele for x in to_process)]

out_dir = "C:/Users/anjonas/PycharmProjects/pytorch-CycleGAN-and-pix2pix/data_predict"

for im in to_process:
    print(im)
    img_name = os.path.basename(im)
    img = imageio.imread(im)
    batches = ["\\1\\", "\\2\\", "\\3\\", "\\4\\", "\\6\\", "\\7\\", "\\11\\"]
    if any(b in im for b in batches):
        type = "dif"
    else:
        type = "dir"
    out_name = img_name.replace(".png", ".jpg")
    imageio.imwrite(f"{out_dir}/{type}/{out_name}", img)

# ======================================================================================================================
# TRAIN THE CYCLE GAN
# ======================================================================================================================

# ======================================================================================================================
# ORDER IMAGES FOR WHICH TO OBTAIN PREDICTIONS (DIF AND DIR)
# ======================================================================================================================

directory = "Z:/Public/Jonas/Data/ESWW006/images_trainset/Output/synthetic_images/[0-9]*"
images = glob.glob(f'{directory}/*.png')
out_dir = "C:/Users/anjonas/PycharmProjects/pytorch-CycleGAN-and-pix2pix/data_predict"

df = pd.DataFrame({'name': images})
df['id'] = utils.get_identifier(images)
df2 = df.groupby(['id']).apply(lambda x: x.sample(10, random_state=10)).reset_index(drop=True)
used = df2['name'].tolist()

for im in used:
    print(im)
    img_name = os.path.basename(im)
    img = imageio.imread(im)
    batches = ["\\1\\", "\\2\\", "\\3\\", "\\4\\", "\\6\\", "\\7\\", "\\11\\"]
    if any(b in im for b in batches):
        type = "dif"
    else:
        type = "dir"
    out_name = img_name.replace(".png", ".jpg")
    imageio.imwrite(f"{out_dir}/{type}/{out_name}", img)

# ======================================================================================================================
# CREATE PREDICTIONS
# ======================================================================================================================

# ======================================================================================================================
# CREATE ORIGINAL IMG / PREDICTION HYBRIDS
# ======================================================================================================================

from PIL import Image, ImageFilter

path_originals = "Z:/Public/Jonas/Data/ESWW006/Images_trainset/Output/cGAN_output/v1"
path_predictions = "Z:/Public/Jonas/Data/ESWW006/Images_trainset/Output/cGAN_output/v1"
path_masks = "Z:/Public/Jonas/Data/ESWW006/images_trainset/Output/annotations_manual/masks/8bit"

masks = glob.glob(f'{path_masks}/*.png')
fn_masks = [os.path.basename(x).replace("_mask.png", "") for x in masks]
originals = glob.glob(f'{path_originals}/*/*_real.png')
fn_originals = utils.get_identifier(originals)
predicted = glob.glob(f'{path_predictions}/*/*_fake.png')
fn_predicted = utils.get_identifier(predicted)

# remove existing ones
existing_composites = glob.glob(f'{path_predictions}/*/*_composite.png')
fn_existing_composites = utils.get_identifier(existing_composites)
fn_originals = [ele for ele in fn_originals if ele not in fn_existing_composites]

for o in np.unique(fn_originals):

    files_o = [i for i in originals if o in i]
    files_p = [i for i in predicted if o in i]
    mask = [i for i in masks if o in i]

    cors = []
    names = []
    for s in files_o:
        real = imageio.imread(s)
        pred = imageio.imread(s.replace("_real", "_fake"))
        hist_real = cv2.calcHist([real], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist_real = cv2.normalize(hist_real, hist_real).flatten()
        hist_pred = cv2.calcHist([pred], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist_pred = cv2.normalize(hist_pred, hist_pred).flatten()
        cor = cv2.compareHist(hist_real, hist_pred, method=cv2.HISTCMP_CORREL)
        cors.append(cor)
        names.append(s)

    index = cors.index(max(cors))
    which = names[index]

    real = imageio.imread(which)
    predicted = imageio.imread(which.replace("_real", "_fake"))
    out = utils.get_plot(which)
    idx = fn_masks.index(out)
    mask = imageio.imread(masks[idx])
    soil_mask = np.bitwise_not(mask)
    soil_mask_3d = np.stack([soil_mask/255, soil_mask/255, soil_mask/255], axis=2)

    # erode vegetation mask
    mask_erode = cv2.erode(mask, np.ones((9, 9), np.uint8))

    predicted_image = Image.fromarray((np.uint8(predicted)))
    real_image = Image.fromarray(np.uint8(real))
    mask_image = Image.fromarray(np.uint8(mask_erode))
    mask_blur = mask_image.filter(ImageFilter.GaussianBlur(5))
    composite = Image.composite(real_image, predicted_image, mask_blur)

    # fig, axs = plt.subplots(1, 4, sharex=True, sharey=True)
    # axs[0].imshow(real_image)
    # axs[0].set_title('real')
    # axs[1].imshow(predicted_image)
    # axs[1].set_title('predictd')
    # axs[2].imshow(composite)
    # axs[2].set_title('composite')
    # axs[3].imshow(mask_blur)
    # axs[3].set_title('mask')
    # plt.show(block=True)

    out_name = which.replace("_real", "_composite")
    imageio.imwrite(out_name, composite)

# ======================================================================================================================
# PREPARE DATA FOR SEGMENTATION TRAINING
# ======================================================================================================================

# ======================================================================================================================
# EITHER: create a new split using all data
# ======================================================================================================================

import random

# process masks first
dir_masks = "Z:/Public/Jonas/Data/ESWW006/images_trainset/Output/annotations_manual/masks"
out_dir_masks = "Z:/Public/Jonas/Data/ESWW006/images_trainset/Output/annotations_manual/masks_patches"

# create train, test, validation from cyclegan output
# list all transformed images
files1 = glob.glob(
    "Z:/Public/Jonas/Data/ESWW006/Images_trainset/Output/cGAN_output/v1/*/*_fake.png")
# add composites
files2 = glob.glob(
    "Z:/Public/Jonas/Data/ESWW006/Images_trainset/Output/cGAN_output/v1/*/*_composite.png")
files = files1 + files2

plots = utils.get_plot_id(files)
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

out_dir = "C:/Users/anjonas/PycharmProjects/SegVeg/data/combined_patches"

# tile and move images and masks
utils.tile_and_move_images(stride=1200,
                           file_list=lst,
                           key_list=lst_key,
                           out_dir=out_dir)

# ======================================================================================================================
# OR: add additional samples to the training data
# ======================================================================================================================

# process masks first
dir_masks = "Z:/Public/Jonas/Data/ESWW006/images_trainset/Output/annotations_manual/masks"
out_dir_masks = "Z:/Public/Jonas/Data/ESWW006/images_trainset/Output/annotations_manual/masks_patches"

# create train, test, validation from cyclegan output
# list all transformed images
files1 = glob.glob(
    "Z:/Public/Jonas/Data/ESWW006/Images_trainset/Output/cGAN_output/v1/*/*_fake.png")
# add composites
files2 = glob.glob(
    "Z:/Public/Jonas/Data/ESWW006/Images_trainset/Output/cGAN_output/v1/*/*_composite.png")
files = files1 + files2

# remove existing
out_dir = "C:/Users/anjonas/PycharmProjects/SegVeg/data/combined_patches"
existing_patches = glob.glob(f"{out_dir}/*/images/*.png")
existing_patches = [os.path.basename(e) for e in existing_patches]
pattern = r'_[0-9].png'
existing_patches = [re.sub(pattern, ".png", e) for e in existing_patches]
ff = [f for f in files if os.path.basename(f) not in existing_patches]
lst_key = ["train" for i in range(len(ff))]

# tile and move images and masks
utils.tile_and_move_images(stride=1200,
                           file_list=ff,
                           key_list=lst_key,
                           out_dir=out_dir)

# ======================================================================================================================
# CREATE DIFFERENT DATA SUBSETS
# ======================================================================================================================

val_dataset = "Z:/Public/Jonas/Data/ESWW006/ImagesNadir/patches_annotation/all_annotations"

# (1) Single tile, single soil, no composites

full_dir = "C:/Users/anjonas/PycharmProjects/SegVeg/data/combined_patches"
all_fakes = glob.glob(f'{full_dir}/*/images/*_fake_[0-9].png')
all_composites = glob.glob(f'{full_dir}/*/images/*_composite_[0-9].png')

df = pd.DataFrame({'name': all_fakes})
df['id'] = utils.get_identifier(all_fakes)
# df['soil'] = [re.sub("_fake_[0-9]", "", utils.get_soil_id(x)) for x in all_fakes]
df2 = df.groupby(['id']).apply(lambda x: x.sample(1, random_state=10)).reset_index(drop=True)
used3 = df2['name'].tolist()

lst, lst_key = utils.split_dataset(files=used3, split_ratio=1.0)
reduced_dir = "C:/Users/anjonas/PycharmProjects/SegVeg/data/1tile_1soil_0composite"
mask_dir = full_dir
utils.tile_and_move_images(stride=None,
                           file_list=lst,
                           key_list=lst_key,
                           out_dir=reduced_dir,
                           validation_dataset_dir=val_dataset)

# ======================================================================================================================

# (2) All four tiles, single soil, no composites

df2 = df.groupby(['id']).apply(lambda x: x.sample(1, random_state=10)).reset_index(drop=True)
used3 = df2['name'].tolist()
used4 = [re.sub("_3.png", "_4.png", x) for x in used3]
used1 = [re.sub("_3.png", "_1.png", x) for x in used3]
used2 = [re.sub("_3.png", "_2.png", x) for x in used3]
used = used1 + used2 + used3 + used4

# WITH COMPOSITES
lst, lst_key = utils.split_dataset(files=used, split_ratio=1.0)
reduced_dir = "C:/Users/anjonas/PycharmProjects/SegVeg/data/4tile_1soil_0composite"
utils.tile_and_move_images(stride=None,
                           file_list=lst,
                           key_list=lst_key,
                           out_dir=reduced_dir,
                           validation_dataset_dir=val_dataset)

# ======================================================================================================================

# (3) All four tiles, n soils, no composites

df2 = df.groupby(['id']).apply(lambda x: x.sample(6, random_state=10)).reset_index(drop=True)
selected = df2['name'].tolist()
all_patches = []
for s in selected:
    trunc = s[:len(s)-5]
    for i in range(1, 5):
        all_patches.append("".join([trunc, str(i), ".png"]))

lst, lst_key = utils.split_dataset(files=all_patches, split_ratio=1.0)
reduced_dir = "C:/Users/anjonas/PycharmProjects/SegVeg/data/4tile_6soil_0composite"
utils.tile_and_move_images(stride=None,
                           file_list=lst,
                           key_list=lst_key,
                           out_dir=reduced_dir,
                           validation_dataset_dir=val_dataset)

# ======================================================================================================================

# (4) All four tiles, n soils, with composites

df2 = df.groupby(['id']).apply(lambda x: x.sample(3, random_state=10)).reset_index(drop=True)
selected = df2['name'].tolist()
all_patches = []
for s in selected:
    trunc = s[:len(s)-5]
    for i in range(1, 5):
        all_patches.append("".join([trunc, str(i), ".png"]))

used = all_patches + all_composites

lst, lst_key = utils.split_dataset(files=used, split_ratio=1.0)
reduced_dir = "C:/Users/anjonas/PycharmProjects/SegVeg/data/4tile_3soil_1composite"
utils.tile_and_move_images(stride=None,
                           file_list=lst,
                           key_list=lst_key,
                           out_dir=reduced_dir,
                           validation_dataset_dir=val_dataset)

# ======================================================================================================================
# TRAIN THE SEGMENTATION MODEL !!
# ======================================================================================================================
#
# ======================================================================================================================
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

# TRAINING

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
    mask_bin = (mask / 255).astype("uint8")
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
    mask_bin = (mask / 255).astype("uint8")
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

# prepare data for pytorch implementation (Zenkl et al. 2022)

dir = "C:/Users/anjonas/PycharmProjects/SegVeg/data/combined"

# copy images
files = glob.glob(f'{dir}/*/images/*.png')
for f in files:
    dirto = f.replace("combined", "combined_pytorch")
    dirto = dirto.replace("\\images", "")
    out_dir = os.path.dirname(dirto)
    if not Path(out_dir).exists():
        Path(out_dir).mkdir(exist_ok=True, parents=True)
    img = imageio.imread(f)
    img_resized = cv2.resize(img, (700, 700), interpolation=cv2.INTER_LINEAR)
    imageio.imwrite(dirto, img_resized)

# reformat and copy masks
files = glob.glob(f'{dir}/*/masks/*.png')
for f in files:
    dirto = f.replace("combined", "combined_pytorch")
    dirto = dirto.replace("\\masks", "")
    dirto = dirto.replace(".png", "_mask.png")
    out_dir = os.path.dirname(dirto)
    if not Path(out_dir).exists():
        Path(out_dir).mkdir(exist_ok=True, parents=True)
    m = imageio.imread(f)
    m_resized = cv2.resize(m, (700, 700), interpolation=cv2.INTER_LINEAR)
    m_out = m_resized * 255
    imageio.imwrite(dirto, m_out)

# ======================================================================================================================

dir = "Z:/Public/Jonas/Data/ESWW006/ImagesNadir/Segmentation"

files = glob.glob(f'{dir}/*.png')
for f in files:
    m = imageio.imread(f)
    # m = m*255
    imageio.imwrite(f.replace(".png", "_mask.png"), m)
