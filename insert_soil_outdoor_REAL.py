import os
import matplotlib as mpl
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

# ======================================================================================================================

# directories
path = "Z:/Public/Jonas/Data/ESWW006/Images_trainset"
out_dir = f"{path}/Output/synthetic_images"
plant_dir = f"{path}/RectPatches"
mask_dir = f"{path}/Output/Mask"

# list soil and plant images
# soils = glob.glob(f'{soil_dir}/*.png')
plants = glob.glob(f'{plant_dir}/*.JPG')
masks = glob.glob(f'{mask_dir}/*.png')

soil_image_type = "Original"  # "Corrected" or "Original"

# iterate over all plants
for p, m in zip(plants, masks):

    stem_name = os.path.basename(p).replace(".JPG", "")
    Plot_ID, date = stem_name.split("_")
    date = "_".join([date[:4], date[4:6], date[6:8]])

    img = imageio.imread(p)
    mask = imageio.imread(m)

    # convert segmentation mask to binary mask
    # mask_0 = np.where(mask == (250, 50, 83), (255, 255, 255), (0, 0, 0))
    # mask_0 = np.uint8(mask_0[:, :, 1])
    mask_0 = np.bitwise_not(mask)  # plant must be foreground

    # erode original mask to get rid of the blueish edge pixels
    kernel = np.ones((2, 2), np.uint8)
    mask_erode = cv2.erode(mask_0, kernel)

    # dilate original mask and invert
    # this will give a soil mask with a "safety margin" along the edges of leaves
    kernel = np.ones((3, 3), np.uint8)
    mask_dilate = cv2.dilate(mask_0, kernel)
    mask_dilate_inv = cv2.bitwise_not(mask_dilate)

    # get connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_dilate)
    _, labels_bg, _, _ = cv2.connectedComponentsWithStats(mask_dilate_inv)

    final_image = copy.copy(img)
    histogram_matching = False
    smoothing = "blur"

    # isolate soil patches in original image, mask plant
    img_ = np.zeros_like(img)
    idx = np.where(labels == 0)
    img_[idx] = img[idx]

    new_soils = glob.glob(f'{path}/{date}/Processed/SoilPatches/{Plot_ID}/{soil_image_type}/*.png')

    # convert the original image with plant masked to gray scale
    soil_gray = cv2.cvtColor(img_, cv2.COLOR_RGB2GRAY)
    soil_gray = soil_gray / soil_gray.max()

    # TODO optimize parameters
    # apply percentile filter sequentially.
    # first using a large kernel to "fill" in soil along long and fairly straight edges
    # this will fill the "gap" between the eroded soil mask and the eroded plant mask
    perc = scipy.ndimage.percentile_filter(soil_gray, percentile=65, size=15, mode='reflect')
    # then using a smaller kernel
    # sequential application to ensure growing of thin or small regions,
    # e.g. small holes in plant or where leaves cross in small angles.
    perc = scipy.ndimage.percentile_filter(perc, percentile=80, size=7, mode='reflect')
    perc = scipy.ndimage.percentile_filter(perc, percentile=80, size=7, mode='reflect')
    perc = scipy.ndimage.percentile_filter(perc, percentile=80, size=7, mode='reflect')
    perc = scipy.ndimage.percentile_filter(perc, percentile=80, size=7, mode='reflect')
    perc = scipy.ndimage.percentile_filter(perc, percentile=80, size=8, mode='reflect')
    perc = scipy.ndimage.percentile_filter(perc, percentile=80, size=8, mode='reflect')
    perc = scipy.ndimage.percentile_filter(perc, percentile=80, size=9, mode='reflect')

    # mask again all plant pixels
    final = np.where(mask_erode == 255, 0, perc)

    # replace all soil pixels in the eroded soil mask with their original intensity values
    ffinal = np.where(labels_bg != 0, soil_gray, final)

    # ==============================================================================================================

    # The above "region growing" does not work for small holes
    # repair by comparing these holes with the holes in the eroded vegetation mask

    # binarize and invert
    bin = np.uint8(np.where(ffinal != 0, 255, 0))
    bin = np.bitwise_not(bin)

    mask_inverted = np.bitwise_not(mask_erode)
    n, lab, st, cent = cv2.connectedComponentsWithStats(mask_inverted)
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
    soil_gray2 = np.where(diff_mask == 255, soil_gray2, ffinal)

    # final post-processing
    final_soil_gray = scipy.ndimage.percentile_filter(np.uint8(soil_gray2 * 255),
                                                      percentile=60, size=7, mode='reflect')
    final_soil_gray = final_soil_gray / 255

    # adjust intensity of each color channel of the new soil image according to the patterns observed
    # on the original soil patch
    for new_soil in new_soils:

        soil_name = os.path.basename(new_soil)
        soil = imageio.imread(new_soil)
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
        blurred_edges = cv2.blur(filled_holes, (3, 3)) * np.dstack([edge_mask, edge_mask, edge_mask])

        # replace "original" edges with blurred edges
        idx = np.where(edge_mask_thin == 1)
        filled_holes[idx] = blurred_edges[idx]

        img_name = f'{stem_name}_{soil_name}'
        snyth_img_path = Path(f'{out_dir}/{soil_image_type}')
        snyth_img_path.mkdir(parents=True, exist_ok=True)

        synth_image = filled_holes

        synth_img_name = snyth_img_path / img_name
        imageio.imwrite(synth_img_name, synth_image)

        # # blur image
        # if smoothing == "blur":
        #     synth_image = cv2.blur(filled_holes, (2, 2))
        #     x, y = synth_image.shape[:2]
        #     x_new = int(np.ceil(x / 4 * 3))
        #     y_new = int(np.ceil(y / 4 * 3))
        #     synth_image = cv2.resize(synth_image, (y_new, x_new), interpolation=cv2.INTER_LINEAR)
        #     mask_res = cv2.resize(mask_0, (y_new, x_new), interpolation=cv2.INTER_NEAREST)

