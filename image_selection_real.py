import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import imageio
import glob
import os
import pandas as pd
import utils
import random
import shutil
from pathlib import Path
import re
import cv2

# ======================================================================================================================
# ESWW006
# ======================================================================================================================

# directories
base_dir = Path("Z:/Public/Jonas/Data/ESWW006/ImagesNadir")
dates = [f for f in os.listdir(base_dir) if re.match(r'[0-9]', f)]
full_dirs = [base_dir / date / "JPEG" for date in dates]
out_dir = base_dir / "patches"
# images with problems, to be removed
meta = pd.read_csv("Z:/Public/Jonas/Data/ESWW006/ImagesNadir/Meta/images_exclude.csv")
problematic_images = meta["Exclude"].tolist()

# iterate over all directories containing intermediate stage canopies and diffuse light
for dir, date in zip(full_dirs, dates):

    print("processing ", date)

    # get all images
    images = glob.glob(f'{dir}/*.JPG')
    images = [item for item in images if "Ref" not in item]  # removes the reference images

    random.seed(10)
    images = random.sample(images, k=30)

    # extract patch from each image
    for i in images:

        img_name = os.path.basename(i)

        # skip if problematic
        if img_name in problematic_images:
            continue

        stem_name = img_name.replace(".JPG", "")

        img = imageio.imread(i)
        ctr_tile = img[0:4200, 2500:7000]

        patch, coords = utils.sample_random_patch(ctr_tile, size=(3000, 3000))
        patch = cv2.resize(patch, (1200, 1200))

        dir = out_dir / "reduced_resolution" / date

        if not dir.exists:
            dir.mkdir(exist_ok=True, parents=True)

        out_name = dir / img_name
        check_dir = dir / "Checker"
        check_name = check_dir / img_name

        # tile if necessary
        # save coordinates and patch(es)
        if not os.path.exists(out_name):
            coordinates = list(coords)
            df = pd.DataFrame([coordinates])
            df2 = df.set_axis(['x1', 'x2', 'y1', 'y2'], axis=1, inplace=False)
            check_dir.mkdir(parents=True, exist_ok=True)
            df2.to_csv(f'{check_dir}/{stem_name}.csv', index=False)
            imageio.imwrite(out_dir / out_name, patch)

# ======================================================================================================================
# FPWW002
# ======================================================================================================================

# directories
base_dir = Path("Z:/Public/Jonas/Data/FPWW002")
dates = os.listdir(base_dir)
dates = dates[0:4] + dates[6:7]
# dates = dates[4:6]
full_dirs = [base_dir / date / "JPEG" for date in dates]
out_dir = base_dir / "patches"
# images with problems, to be removed
meta = pd.read_csv("Z:/Public/Jonas/Data/ESWW006/ImagesNadir/Meta/images_exclude.csv")
problematic_images = meta["Exclude"].tolist()

# iterate over all directories containing intermediate stage canopies and diffuse light
for dir, date in zip(full_dirs, dates):

    print(date)

    # get all images
    images = glob.glob(f'{dir}/*.JPG')
    images = [item for item in images if "Ref" not in item]  # removes the reference images

    random.seed(10)
    images = random.sample(images, k=50)

    # extract patch from each image
    for i in images:

        img_name = os.path.basename(i)

        # skip if problematic
        if img_name in problematic_images:
            continue

        stem_name = img_name.replace(".JPG", "")
        png_name = img_name.replace(".JPG", ".png")

        img = imageio.imread(i)
        ctr_tile = img[0:2500, 1800:4200]

        patch, coords = utils.sample_random_patch(ctr_tile, size=(1200, 1200))

        out_name = out_dir / date / img_name
        check_dir = out_dir / date / "Checker"
        check_name = check_dir / img_name

        # tile if necessary
        # save coordinates and patch(es)
        if not os.path.exists(out_name):
            coordinates = list(coords)
            df = pd.DataFrame([coordinates])
            df2 = df.set_axis(['x1', 'x2', 'y1', 'y2'], axis=1, inplace=False)
            check_dir.mkdir(parents=True, exist_ok=True)
            df2.to_csv(f'{check_dir}/{stem_name}.csv', index=False)
            imageio.imwrite(out_dir / out_name, patch, quality=100)

# ======================================================================================================================

# split data set and move to training/testing folder

base_dir = Path("Z:/Public/Jonas/Data/ESWW006/ImagesNadir")
from_dir = base_dir / "patches" / "[0-9]*"
to_dir = "Z:/Public/Jonas/Data/ESWW006/images_trainset/Output/CGAN_input/composite2real_int"
images_ESWW006 = glob.glob(f'{from_dir}/*.JPG')

# # sample a subset of images
# random.seed(10)
# images_ESWW006 = random.sample(images_ESWW006, k=round(len(images_ESWW006) * 0.4))
# get only intermediate
intermed = pd.read_csv("Z:/Public/Jonas/Data/ESWW006/ImagesNadir/Meta/phenology.csv")
intermed = intermed[intermed['pheno'] == "int"]['image_ID'].tolist()
images_ESWW006 = [x for x in images_ESWW006 if os.path.basename(x) in intermed]

base_dir = Path("Z:/Public/Jonas/Data/FPWW002")
from_dir = base_dir / "patches" / "*"
images_FPWW002 = glob.glob(f'{from_dir}/*.JPG')

images = images_ESWW006 + images_FPWW002

random.seed(10)
trainA = random.sample(images, k=round(len(images)*0.8))
testA = [item for item in images if item not in trainA]

for im in testA:
    im_name = os.path.basename(im)
    out_name = im_name.replace(".JPG", ".jpg")
    dst_dir = f"{to_dir}/testA/{out_name}"
    shutil.copy(im, dst_dir)

# ======================================================================================================================
# ======================================================================================================================

# ======================================================================================================================
# (1) Int/dif and int/int
# ======================================================================================================================

# directories
type = "int_dif"
size = "small"  # "large" for 2400 x 2400, "small" for 1200 x 1200
base_dir = Path("Z:/Public/Jonas/Data/ESWW006/ImagesNadir")
# dirs = ["2022_06_13/JPEG", "2022_06_22/JPEG", "2022_06_27/JPEG",
#         "2022_06_30/JPEG/subset_stg", "2022_07_04/JPEG/subset_stg"]
dirs = ["2022_06_13/JPEG"]
full_dirs = [base_dir / x for x in dirs]
# out_dir = base_dir / "patches" / size / type
out_dir = base_dir / "patches" / type
check_dir = out_dir / "Checker"
# images with problems, to be removed
meta = pd.read_csv("Z:/Public/Jonas/Data/ESWW006/ImagesNadir/Meta/images_exclude.csv")
problematic_images = meta["Exclude"].tolist()

# iterate over all directories containing intermediate stage canopies and diffuse light
for d in full_dirs:

    # get all images
    images = glob.glob(f'{d}/*.JPG')
    images = [item for item in images if "Ref" not in item]  # removes the reference images

    # extract patch from each image
    for i in images:

        img_name = os.path.basename(i)

        # skip if problematic
        if img_name in problematic_images:
            continue

        stem_name = img_name.replace(".JPG", "")

        img = imageio.imread(i)
        ctr_tile = img[0:4200, 2500:7000]

        patch, coords = utils.sample_random_patch(ctr_tile, size=(2400, 2400))

        out_name = out_dir / img_name
        check_name = check_dir / img_name

        # tile if necessary
        # save coordinates and patch(es)
        if not os.path.exists(out_name):
            coordinates = list(coords)
            df = pd.DataFrame([coordinates])
            df2 = df.set_axis(['x1', 'x2', 'y1', 'y2'], axis=1, inplace=False)
            check_dir.mkdir(parents=True, exist_ok=True)
            df2.to_csv(f'{check_dir}/{stem_name}.csv', index=False)
            if size == "small":
                tiles = utils.image_tiler(patch, stride=1200)
                for i, tile in enumerate(tiles):
                    out_name = img_name.replace(".JPG", f"_{i + 1}.JPG")
                    imageio.imwrite(out_dir / out_name, tile)
            elif size == "large":
                imageio.imwrite(out_dir / img_name, patch)

# ======================================================================================================================

# split data set and move to training/testing folder

to_dir = "Z:/Public/Jonas/Data/ESWW006/images_trainset/Output/CGAN_input/composite2real_int_dif"

images = glob.glob(f'{out_dir}/*.JPG')

trainA = random.sample(images, k=round(len(images)*0.8))
testA = [item for item in images if item not in trainA]

for im in testA:
    im_name = os.path.basename(im)
    out_name = im_name.replace(".JPG", ".jpg")
    dst_dir = f"{to_dir}/testA/{out_name}"
    shutil.copy(im, dst_dir)

# ======================================================================================================================
# (2) Int/dir
# ======================================================================================================================

# directories
type = "int_dir"
size = "large"  # "large" for 2400 x 2400, "small" for 1200 x 1200
base_dir = Path("Z:/Public/Jonas/Data/ESWW006/ImagesNadir")
dirs = ["2022_06_20/JPEG", "2022_06_23/JPEG", "2022_06_25/JPEG",
        "2022_06_30/JPEG/subset_stg", "2022_07_04/JPEG/subset_stg"]
full_dirs = [base_dir / x for x in dirs]
extra_dirs = ["Z:/Public/Jonas/Data/FPWW002/2013-07-16_WW002_037-108",
              "Z:/Public/Jonas/Data/FPWW002/2013-07-19_WW002_037-108"]
full_dirs = full_dirs + extra_dirs
out_dir = base_dir / "patches" / size / type
check_dir = out_dir / "Checker"
# images with problems, to be removed
meta = pd.read_csv("Z:/Public/Jonas/Data/ESWW006/ImagesNadir/Meta/images_exclude.csv")
problematic_images = meta["Exclude"].tolist()

# iterate over all directories containing intermediate stage canopies and diffuse light
for d in full_dirs:

    # get all images
    images = glob.glob(f'{d}/*.JPG')
    images = [item for item in images if "Ref" not in item]  # removes the reference images

    # extract patch from each image
    for i in images:

        img_name = os.path.basename(i)

        # skip if problematic
        if img_name in problematic_images:
            continue

        stem_name = img_name.replace(".JPG", "")

        img = imageio.imread(i)
        ctr_tile = img[0:4200, 2500:7000]

        patch, coords = utils.sample_random_patch(ctr_tile, size=(2400, 2400))

        out_name = out_dir / img_name
        check_name = check_dir / img_name

        # tile if necessary
        # save coordinates and patch(es)
        if not os.path.exists(out_name):
            coordinates = list(coords)
            df = pd.DataFrame([coordinates])
            df2 = df.set_axis(['x1', 'x2', 'y1', 'y2'], axis=1, inplace=False)
            check_dir.mkdir(parents=True, exist_ok=True)
            df2.to_csv(f'{check_dir}/{stem_name}.csv', index=False)
            if size == "small":
                tiles = utils.image_tiler(patch, stride=1200)
                for i, tile in enumerate(tiles):
                    out_name = img_name.replace(".JPG", f"_{i + 1}.JPG")
                    imageio.imwrite(out_dir / out_name, tile)
            elif size == "large":
                imageio.imwrite(out_dir / img_name, patch)

# # ======================================================================================================================
#
# # split data set and move to training/testing folder
#
# to_dir = "Z:/Public/Jonas/Data/ESWW006/images_trainset/Output/CGAN_input/composite2real_int_dir"
#
# images = glob.glob(f'{out_dir}/*.JPG')
#
# trainA = random.sample(images, k=round(len(images)*0.8))
# testA = [item for item in images if item not in trainA]
#
# for im in testA:
#     im_name = os.path.basename(im)
#     print(im_name)
#     out_name = im_name.replace(".JPG", ".jpg")
#     dst_dir = f"{to_dir}/testA/{out_name}"
#     shutil.copy(im, dst_dir)
#
# # ======================================================================================================================

# ======================================================================================================================
# (2) stg/dir
# ======================================================================================================================

# directories
type = "stg_dir"
size = "large"  # "large" for 2400 x 2400, "small" for 1200 x 1200
base_dir = Path("Z:/Public/Jonas/Data/ESWW006/ImagesNadir")
dirs = ["2022_05_25/JPEG", "2022_05_28/JPEG", "2022_05_30/JPEG",
        "2022_06_04/JPEG", "2022_06_15/JPEG", "2022_06_17/JPEG"]
full_dirs = [base_dir / x for x in dirs]
out_dir = base_dir / "patches" / size / type
check_dir = out_dir / "Checker"
# images with problems, to be removed
meta = pd.read_csv("Z:/Public/Jonas/Data/ESWW006/ImagesNadir/Meta/images_exclude.csv")
problematic_images = meta["Exclude"].tolist()

# iterate over all directories containing intermediate stage canopies and diffuse light
for d in full_dirs:

    # get all images
    images = glob.glob(f'{d}/*.JPG')
    images = [item for item in images if "Ref" not in item]  # removes the reference images

    # extract patch from each image
    for i in images:

        img_name = os.path.basename(i)

        # skip if problematic
        if img_name in problematic_images:
            continue

        stem_name = img_name.replace(".JPG", "")

        img = imageio.imread(i)
        ctr_tile = img[0:4200, 2500:7000]

        patch, coords = utils.sample_random_patch(ctr_tile, size=(2400, 2400))

        out_name = out_dir / img_name
        check_name = check_dir / img_name

        # tile if necessary
        # save coordinates and patch(es)
        if not os.path.exists(out_name):
            coordinates = list(coords)
            df = pd.DataFrame([coordinates])
            df2 = df.set_axis(['x1', 'x2', 'y1', 'y2'], axis=1, inplace=False)
            check_dir.mkdir(parents=True, exist_ok=True)
            df2.to_csv(f'{check_dir}/{stem_name}.csv', index=False)
            if size == "small":
                tiles = utils.image_tiler(patch, stride=1200)
                for i, tile in enumerate(tiles):
                    out_name = img_name.replace(".JPG", f"_{i + 1}.JPG")
                    imageio.imwrite(out_dir / out_name, tile)
            elif size == "large":
                imageio.imwrite(out_dir / img_name, patch)

# ======================================================================================================================
# (2) stg/dif
# ======================================================================================================================

# directories
type = "stg_dif"
size = "large"  # "large" for 2400 x 2400, "small" for 1200 x 1200
base_dir = Path("Z:/Public/Jonas/Data/ESWW006/ImagesNadir")
dirs = ["2022_06_01/JPEG", "2022_06_13/JPEG"]
full_dirs = [base_dir / x for x in dirs]
out_dir = base_dir / "patches" / size / type
check_dir = out_dir / "Checker"
# images with problems, to be removed
meta = pd.read_csv("Z:/Public/Jonas/Data/ESWW006/ImagesNadir/Meta/images_exclude.csv")
problematic_images = meta["Exclude"].tolist()

# iterate over all directories containing intermediate stage canopies and diffuse light
for d in full_dirs:

    # get all images
    images = glob.glob(f'{d}/*.JPG')
    images = [item for item in images if "Ref" not in item]  # removes the reference images

    # extract patch from each image
    for i in images:

        img_name = os.path.basename(i)

        # skip if problematic
        if img_name in problematic_images:
            continue

        stem_name = img_name.replace(".JPG", "")

        img = imageio.imread(i)
        ctr_tile = img[0:4200, 2500:7000]

        patch, coords = utils.sample_random_patch(ctr_tile, size=(2400, 2400))

        out_name = out_dir / img_name
        check_name = check_dir / img_name

        # tile if necessary
        # save coordinates and patch(es)
        if not os.path.exists(out_name):
            coordinates = list(coords)
            df = pd.DataFrame([coordinates])
            df2 = df.set_axis(['x1', 'x2', 'y1', 'y2'], axis=1, inplace=False)
            check_dir.mkdir(parents=True, exist_ok=True)
            df2.to_csv(f'{check_dir}/{stem_name}.csv', index=False)
            if size == "small":
                tiles = utils.image_tiler(patch, stride=1200)
                for i, tile in enumerate(tiles):
                    out_name = img_name.replace(".JPG", f"_{i + 1}.JPG")
                    imageio.imwrite(out_dir / out_name, tile)
            elif size == "large":
                imageio.imwrite(out_dir / img_name, patch)

# ======================================================================================================================
# FPWW002 --> ESWW006
# ======================================================================================================================

# source domain FPWW002
dirA = Path("Z:/Public/Jonas/Data/FPWW002/patches")
imagesA = glob.glob(f'{dirA}/2013*/*.JPG')

# target domain ESWW006
dirB = Path("Z:/Public/Jonas/Data/ESWW006/ImagesNadir/patches/reduced_resolution")
imagesB = glob.glob(f'{dirB}/[0-9]*/*.JPG')

to_dir = "Z:/Public/Jonas/Data/FPWW002/domain_transfer"

trainA = random.sample(imagesA, k=round(len(imagesA)*0.75))
testA = [item for item in imagesA if item not in trainA]

trainB = random.sample(imagesB, k=round(len(imagesB)*0.75))
testB = [item for item in imagesB if item not in trainB]

for im in trainB:
    im_name = os.path.basename(im)
    out_name = im_name.replace(".JPG", ".jpg")
    dst_dir = f"{to_dir}/trainB/{out_name}"
    shutil.copy(im, dst_dir)