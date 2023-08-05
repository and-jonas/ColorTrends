
import glob
import os
import imageio
import numpy as np

# ======================================================================================================================
# 1. Gather all annotations from ESWW006, ESWW007, and FPWW002
# ======================================================================================================================

all_annotated = glob.glob("P:/Public/Jonas/Data/ESWW006/ImagesNadir/patches_annotation_sideview/All/*/SegmentationClass/*.png")
# path_coordinates = "Z:/Public/Jonas/Data/ESWW006/ImagesNadir/patches_annotation_stems/images_ESWW007_ESWW008/"
# path_coordinates_patch = "Z:/Public/Jonas/Data/ESWW006/ImagesNadir/patches_annotation_spikes/coordinates"
path_original_images1 = "Z:/Public/Jonas/Data/ESWW006/ImagesNadir/patches_annotation_stems/images_ESWW007_ESWW008/608px"
path_original_images2 = "Z:/Public/Jonas/Data/ESWW006/ImagesNadir/patches_annotation_stems/images_ESWW007_ESWW008/1200px"
path_proc = "P:/Public/Jonas/Data/ESWW006/ImagesNadir/patches_annotation_sideview/processed"

for ann in all_annotated:
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
    try:
        new_img = imageio.imread(f'{path_original_images1}/{base_name}')
    except FileNotFoundError:
        new_img = imageio.imread(f'{path_original_images2}/{base_name}')
    # get mask
    mask = imageio.imread(ann)
    mask_ = np.zeros((mask.shape[0], mask.shape[1]))
    idx1 = np.where(mask == (51, 221, 255))[:2]
    mask_[idx1] = 1
    idx2 = np.where(mask == (250, 50, 83))[:2]
    mask_[idx2] = 2
    idx3 = np.where(mask == (221, 255, 51))[:2]
    mask_[idx3] = 0
    mask_checker = np.uint8(mask_ * 255/2)
    imageio.imwrite(f'{path_proc}/{mask_name}', mask)
    # checker
    imageio.imwrite(f'{path_proc}/checker/{mask_name}', mask_checker)
    imageio.imwrite(f'{path_proc}/checker/{base_name}', new_img)
