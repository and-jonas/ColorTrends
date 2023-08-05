
import numpy as np
from PIL import Image
import glob
import os
import re
import random
import pandas as pd
from pathlib import Path
import imageio
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

workdir = 'Z:/Public/Jonas/Data/ESWW006/ImagesNadir'
dir_patch_coordinates = f'{workdir}/Meta/patch_coordinates'
dir_ear_mask = f'{workdir}/Output/SegEar/Mask'
dir_veg_mask = f'{workdir}/Output/SegVeg/Mask'
out_dir = "Z:/Public/Jonas/003_ESWW/Manuscript/Figures/Figure3/overlays"

dirs = [f for f in os.listdir(workdir) if re.match('^2022_', f)]
# dirs = [os.path.join(workdir, d) for d in dirs if d  not in ['2022_06_01', '2022_06_23', '2022_05_28']]
dirs = [os.path.join(workdir, d) for d in dirs]
dirs = [d + "/JPEG" for d in dirs]

files = []
for d in dirs:
    f = glob.glob(f'{d}/*.JPG')
    files = [f for f in files if "Ref" not in f]
    files.append(random.choice(f))

for f in files:
    base_name = os.path.basename(f)
    stem_name = Path(f).stem
    png_name = base_name.replace("." + "JPG", ".png")
    img = Image.open(f)
    pix = np.array(img)

    # sample patch from image
    c = pd.read_table(f'{dir_patch_coordinates}/{stem_name}.txt', sep=",").iloc[0, :].tolist()
    patch = pix[c[2]:c[3], c[0]:c[1]]

    # make overlay
    mask_8bit_ear = imageio.imread(f'{dir_ear_mask}/{png_name}')
    mask_8bit_veg = imageio.imread(f'{dir_veg_mask}/{png_name}')

    M_ear = mask_8bit_ear.ravel()
    M_ear = np.expand_dims(M_ear, -1)
    out_mask = np.dot(M_ear, np.array([[1, 0, 0, 0.33]]))
    out_mask = np.reshape(out_mask, newshape=(patch.shape[0], patch.shape[1], 4))
    out_mask = out_mask.astype("uint8")
    mask = Image.fromarray(out_mask, mode="RGBA")
    img_ = Image.fromarray(patch, mode="RGB")
    img_ = img_.convert("RGBA")
    img_.paste(mask, (0, 0), mask)
    img_ = img_.convert('RGB')
    overlay = np.asarray(img_)

    M_veg = mask_8bit_veg.ravel()
    M_veg = np.expand_dims(M_veg, -1)
    out_mask = np.dot(M_veg, np.array([[0, 0, 1, 0.33]]))
    out_mask = np.reshape(out_mask, newshape=(patch.shape[0], patch.shape[1], 4))
    out_mask = out_mask.astype("uint8")
    mask = Image.fromarray(out_mask, mode="RGBA")
    img_ = Image.fromarray(overlay, mode="RGB")
    img_ = img_.convert("RGBA")
    img_.paste(mask, (0, 0), mask)
    img_ = img_.convert('RGB')
    overlay = np.asarray(img_)

    out_image = np.zeros_like(patch)
    out_image[:1500, :, :] = patch[:1500, :, :]
    out_image[1500:, :, :] = overlay[1500:, :, :]

    out_name = f'{out_dir}/{png_name}'

    imageio.imwrite(out_name, out_image)

# ======================================================================================================================

workdir = 'Z:/Public/Jonas/Data/ESWW006/ImagesNadir'
dir_patch_coordinates = f'{workdir}/Meta/patch_coordinates'
out_dir = "Z:/Public/Jonas/003_ESWW/Manuscript/Figures/Figure5/patches"

dirs = [f for f in os.listdir(workdir) if re.match('^2022_', f)]
# dirs = [os.path.join(workdir, d) for d in dirs if d  not in ['2022_06_01', '2022_06_23', '2022_05_28']]
dirs = [os.path.join(workdir, d) for d in dirs]
dirs = [d + "/JPEG" for d in dirs]

files = []
for d in dirs:
    f = glob.glob(f'{d}/*.JPG')
    files = [f for f in files if "Ref" not in f]
    files.append(random.choice(f))

for f in files:
    base_name = os.path.basename(f)
    stem_name = Path(f).stem
    png_name = base_name.replace("." + "JPG", ".png")
    img = Image.open(f)
    pix = np.array(img)

    # sample patch from image
    c = pd.read_table(f'{dir_patch_coordinates}/{stem_name}.txt', sep=",").iloc[0, :].tolist()
    patch = pix[c[2]:c[3], c[0]:c[1]]

    out_name = f'{out_dir}/{png_name}'

    imageio.imwrite(out_name, patch)

# ======================================================================================================================

out_dir = "Z:/Public/Jonas/003_ESWW/Documentation/Figures_talk"

files = []
for d in dirs:
    f = glob.glob(f'{d}/*ESWW0060050_Cnp_1.JPG')
    files = [f for f in files if "Ref" not in f]
    files.append(random.choice(f))

for f in files:
    base_name = os.path.basename(f)
    stem_name = Path(f).stem
    png_name = base_name.replace("." + "JPG", ".png")
    img = Image.open(f)
    pix = np.array(img)

    # sample patch from image
    c = pd.read_table(f'{dir_patch_coordinates}/{stem_name}.txt', sep=",").iloc[0, :].tolist()
    patch = pix[c[2]:c[3], c[0]:c[1]]

    # make overlay
    mask_8bit_ear = imageio.imread(f'{dir_ear_mask}/{png_name}')
    mask_8bit_veg = imageio.imread(f'{dir_veg_mask}/{png_name}')

    M_ear = mask_8bit_ear.ravel()
    M_ear = np.expand_dims(M_ear, -1)
    out_mask = np.dot(M_ear, np.array([[1, 0, 0, 0.33]]))
    out_mask = np.reshape(out_mask, newshape=(patch.shape[0], patch.shape[1], 4))
    out_mask = out_mask.astype("uint8")
    mask = Image.fromarray(out_mask, mode="RGBA")
    img_ = Image.fromarray(patch, mode="RGB")
    img_ = img_.convert("RGBA")
    img_.paste(mask, (0, 0), mask)
    img_ = img_.convert('RGB')
    overlay = np.asarray(img_)

    M_veg = mask_8bit_veg.ravel()
    M_veg = np.expand_dims(M_veg, -1)
    out_mask = np.dot(M_veg, np.array([[0, 0, 1, 0.33]]))
    out_mask = np.reshape(out_mask, newshape=(patch.shape[0], patch.shape[1], 4))
    out_mask = out_mask.astype("uint8")
    mask = Image.fromarray(out_mask, mode="RGBA")
    img_ = Image.fromarray(overlay, mode="RGB")
    img_ = img_.convert("RGBA")
    img_.paste(mask, (0, 0), mask)
    img_ = img_.convert('RGB')
    overlay = np.asarray(img_)

    out_image = np.zeros_like(patch)
    out_image[:1500, :, :] = patch[:1500, :, :]
    out_image[1500:, :, :] = overlay[1500:, :, :]

    out_name = f'{out_dir}/{png_name}'

    imageio.imwrite(out_name, out_image)
