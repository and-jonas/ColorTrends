import glob
import pandas as pd
import numpy as np
import os
import imageio
import shutil
from pathlib import Path
import utils

data_from = "Z:/Public/Jonas/Data/ESWW006/ImagesNadir/patches_annotation"
data_to = "Z:/Public/Jonas/Data/ESWW006/ImagesNadir/patches_annotation/all_annotations"

images = glob.glob(f'{data_from}/*/*.png')
masks = glob.glob(f'{data_from}/*/SegmentationClass/*.png')
edited = glob.glob(f'{data_from}/*/edited.txt')

out = []
for e in edited:
    ed = pd.read_csv(e, header=None)
    imgs = ed.iloc[:, 0].tolist()
    out.extend(imgs)

validation_samples = [item for item in images if os.path.basename(item) in out]
validation_targets = [item for item in masks if os.path.basename(item) in out]

masks_out_dir = f"{data_to}/masks"
masks_out_dir_8bit = f"{data_to}/masks/8bit"
# create directories
Path(masks_out_dir).mkdir(exist_ok=True, parents=True)
Path(masks_out_dir_8bit).mkdir(exist_ok=True, parents=True)

for s in validation_samples:
    base_name = os.path.basename(s)
    shutil.copy(s, f'{data_to}/images/{base_name}')
    shutil.copy(s, f"{masks_out_dir_8bit}/{base_name}")

for t in validation_targets:
    base_name = os.path.basename(t)
    stem_name = base_name.replace(".png", "")
    mask = imageio.imread(t)
    # check if mask is binary; binarize if needed
    if not len(mask.shape) == 2:
        mask = utils.binarize_mask(mask)
    # save
    imageio.imwrite(f"{masks_out_dir}/{stem_name}.png", np.uint8(mask / 255))
    imageio.imwrite(f"{masks_out_dir_8bit}/{stem_name}_mask.png", mask)
