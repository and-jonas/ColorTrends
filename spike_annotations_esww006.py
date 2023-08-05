import glob
import pandas as pd
import os
import imageio
import utils
import cv2

# ======================================================================================================================
# ESWW006
# ======================================================================================================================

# get a list of relevant files, selected previously
# see: Z:/Public/Jonas/003_ESWW/RScripts/image_selection_annotation.R
out_dir = "Z:/Public/Jonas/Data/ESWW006/ImagesNadir"
selected = pd.read_csv(f'{out_dir}/Meta/selected_for_spike_annotations.csv')
# selected = selected.iloc[16:]
plots = selected["image_ID"].tolist()
full_names_all = glob.glob(f'{out_dir}/2022_*/JPEG/*.JPG')
base_names_all = [os.path.basename(x) for x in full_names_all]

# Indices list of matching element from other list
res = []
i = 0
while i < len(base_names_all):
    if plots.count(base_names_all[i]) > 0:
        res.append(i)
    i += 1

full_names_sel = [full_names_all[i] for i in res]

# FOR EACH IMAGE EXTRACT A PATCH OF 2700 X 2700
path_coordinates = "Z:/Public/Jonas/Data/ESWW006/ImagesNadir/Meta/patch_coordinates"
for image in full_names_sel:
    base_name = os.path.basename(image)
    stem_name = base_name.replace(".JPG", "")
    txt_name = f'{stem_name}.txt'
    png_name = f'{stem_name}.png'
    rc = tuple(pd.read_csv(f'{path_coordinates}/{txt_name}').iloc[0])
    img = imageio.imread(image)
    new_img = img[rc[2]:rc[3], rc[0]:rc[1]]
    patch, c = utils.sample_random_patch(new_img, (2700, 2700))
    # resize and save checker
    checker_name = f'{out_dir}/patches_annotation/checkers/{base_name}'
    x_new = int(patch.shape[0]*(1/2.25))
    y_new = int(patch.shape[1]*(1/2.25))
    patch = cv2.resize(patch, (y_new, x_new), interpolation=cv2.INTER_LINEAR)
    # save patch
    imageio.imwrite(f'{out_dir}/patches_annotation_spikes/{stem_name}.png', patch)
    # save coordinates
    df = pd.DataFrame([c])
    df2 = df.set_axis(['x1', 'x2', 'y1', 'y2'], axis=1, inplace=False)
    df2.to_csv(f'{out_dir}/patches_annotation_spikes/coordinates/{stem_name}.txt', index=False)

# ======================================================================================================================
# ESWW007
# ======================================================================================================================

full_names_sel = glob.glob("Z:/Public/Jonas/Data/ESWW006/ImagesNadir/patches_annotation_stems/images_ESWW007_ESWW008/Images/Nadir/*.JPG")
out_dir = "Z:/Public/Jonas/Data/ESWW006/ImagesNadir"

# FOR EACH IMAGE EXTRACT A PATCH OF 2700 X 2700
path_coordinates = "Z:/Public/Jonas/Data/ESWW006/ImagesNadir/patches_annotation_stems/images_ESWW007_ESWW008/2700px/coordinates"

for image in full_names_sel:
    base_name = os.path.basename(image)
    stem_name = base_name.replace(".JPG", "")
    txt_name = f'{stem_name}.txt'
    png_name = f'{stem_name}.png'
    rc = tuple(pd.read_csv(f'{path_coordinates}/{txt_name}').iloc[0])
    img = imageio.imread(image)
    new_img = img[rc[2]:rc[3], rc[0]:rc[1]]
    patch, c = utils.sample_random_patch(new_img, (2700, 2700))
    # resize and save checker
    checker_name = f'{out_dir}/patches_annotation/checkers/{base_name}'
    # x_new = int(patch.shape[0]*(1/2.25))
    # y_new = int(patch.shape[1]*(1/2.25))
    # patch = cv2.resize(patch, (y_new, x_new), interpolation=cv2.INTER_LINEAR)
    # save patch
    imageio.imwrite(f'{out_dir}/patches_annotation_spikes/{stem_name}.png', patch)
    # save coordinates
    df = pd.DataFrame([c])
    df2 = df.set_axis(['x1', 'x2', 'y1', 'y2'], axis=1, inplace=False)
    df2.to_csv(f'{out_dir}/patches_annotation_spikes/coordinates/{stem_name}.txt', index=False)
