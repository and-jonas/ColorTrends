import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import imageio
import numpy as np
import cv2
import utils
from skimage import exposure
import PIL
import copy
from importlib import reload
reload(utils)
import glob

# ======================================================================================================================

example = imageio.imread("P:/Public/Jonas/003_ESWW/ColorTrends/TrainSoil/large_soil/soil_patches/FPWW0120077_RGB1_20160701_112342_4.png")
ex_gray = cv2.cvtColor(example, cv2.COLOR_RGB2GRAY)
ex_gray = ex_gray/ex_gray.max()
plt.imshow(ex_gray)

# left = ex_gray[:, :104]
img_right = example[:, 104:, :3]
img_left = example[:, :104, :3]
gray_right = ex_gray[:, 104:]
# pseudo_col = np.stack([gray_right, gray_right, gray_right], axis=2)
r, g, b = cv2.split(img_left)
r_ = r * gray_right
g_ = g * gray_right
b_ = b * gray_right
img_ = cv2.merge((r_, g_, b_))
corr_factor = np.mean(img_right)/np.mean(img_)
img_ = img_ * corr_factor
# rescale to original range

matched = exposure.match_histograms(img_, img_right, multichannel=True)
img_ = np.uint8(matched)
plt.imshow(img_)

fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
axs[0].imshow(img_right)
axs[0].set_title('original')
axs[1].imshow(img_left)
axs[1].set_title('soil patch')
axs[2].imshow(img_)
axs[2].set_title('fake')
plt.show(block=True)

# ======================================================================================================================

mask = imageio.imread("Z:/Public/Jonas/003_ESWW/ColorTrends/TrainSoil/large_soil/test/SegmentationClass/train_IMG_1171.png")
img = imageio.imread("Z:/Public/Jonas/003_ESWW/ColorTrends/TrainingSpikes/train_IMG_1171.png")
plt.imshow(mask)

soil_pixels = np.where(mask == (229, 82, 74))
img_blue = copy.copy(img)
img_blue[soil_pixels[0], soil_pixels[1], ] = (0, 0, 255)
# soil = imageio.imread("P:/Public/Jonas/003_ESWW/ColorTrends/TrainSoil/large_soil_new/soil_patches/FPWW0320416_RGB1_20210409_091907_1.png")
# soil = imageio.imread("P:/Public/Jonas/003_ESWW/ColorTrends/TrainSoil/large_soil_new/soil_patches/FPWW0320084_RGB1_20210531_103354_1.png")

out_dir = "P:/Public/Jonas/003_ESWW/ColorTrends/TrainSoil/large_soil_new/synthetic_images"

soil_dir = "P:/Public/Jonas/003_ESWW/ColorTrends/TrainSoil/large_soil_new/soil_patches/"
soils = glob.glob(f'{soil_dir}/*.png')

mask_ = np.where(mask == (229, 82, 74), mask, (0, 0, 0))
mask_ = np.where(mask == (229, 82, 74), (255, 255, 255), (0, 0, 0))
mask_ = mask_[:, :, 1]
mask_ = np.uint8(mask_)
mask_ = utils.filter_objects_size(mask_, 20, "smaller")

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_)
# plt.imshow(labels)

final_image = copy.copy(img)

histogram_matching = True
smoothing = "original"

for i, s in enumerate(soils):

    soil = imageio.imread(s)

    for label in np.unique(labels)[1:]:

        # isolate soil patch, mask rest of image
        img_ = np.zeros_like(img)
        idx = np.where(labels == label)
        img_[idx] = img[idx]

        # get bounding box
        min_x, max_x, min_y, max_y = (np.min(idx[0]), np.max(idx[0]), np.min(idx[1]), np.max(idx[1]))
        original_soil = img_[min_x:max_x, min_y:max_y]
        original_soil_full_patch = final_image[min_x:max_x, min_y:max_y]

        # OPTIONAL: resize the new soil image
        size_x, size_y = original_soil.shape[:2]
        size_x_s, size_y_s = soil.shape[:2]
        size_x_f = size_x/size_x_s
        size_y_f = size_y/size_y_s
        size_f = np.max([size_x_f, size_y_f, 0.75])
        new_soil = cv2.resize(soil, (int(size_f*size_y_s + 1), int(size_f*size_x_s + 1)))
        new_soil, _ = utils.random_soil_patch(img=new_soil, size=(size_x, size_y))

        # convert to gray scale and average intensities in sunlit and shaded parts
        # MODIFY ONCE THE ACTUAL INTENSITIES ON ARTIFICIAL BACKGROUND ARE AVAILABLE
        soil_gray = cv2.cvtColor(original_soil, cv2.COLOR_RGB2GRAY)
        soil_gray = soil_gray/soil_gray.max()
        shade_mask = np.where(soil_gray > 0.35, 0, 1)
        light_mask = np.where(soil_gray < 0.35, 0, 1)
        patch_mask = np.where(soil_gray != 0, 255, 0)
        MASK_SHADE = shade_mask * patch_mask
        MASK_LIGHT = light_mask * patch_mask
        idx_shade = np.where(MASK_SHADE != 0)
        idx_light = np.where(MASK_LIGHT != 0)
        mean_shade = np.mean(soil_gray[idx_shade])
        mean_light = np.mean(soil_gray[idx_light])
        soil_gray_ = copy.copy(soil_gray)
        soil_gray_[idx_shade] = mean_shade
        soil_gray_[idx_light] = mean_light
        soil_gray_blur = cv2.blur(soil_gray_, (7, 7))

        # adjust intensity of each color channel of the new soil image according to the patterns observed
        # on the original soil patch
        r, g, b = cv2.split(new_soil[:, :, :3])
        r_ = r * soil_gray_blur
        g_ = g * soil_gray_blur
        b_ = b * soil_gray_blur
        img_ = cv2.merge((r_, g_, b_))
        img_ = np.uint8(img_)
        corr_factor = np.mean(original_soil)/np.mean(img_)
        img_ = img_ * corr_factor

        # match intensity histograms for the original and the new soil patch
        if histogram_matching:
            matched = exposure.match_histograms(img_, original_soil, multichannel=True)
        else:
            matched = img_
        img_ = np.uint8(matched)

        # fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
        # axs[0].imshow(original_soil)
        # axs[0].set_title('original')
        # axs[1].imshow(new_soil)
        # axs[1].set_title('soil patch')
        # axs[2].imshow(img_blurred)
        # axs[2].set_title('fake')
        # plt.show(block=True)

        # insert patch
        transparency = np.ones_like(img_[:, :, 0])*255
        transparency[np.where(soil_gray_ == 0)] = 0
        img_final = np.dstack([img_, transparency])
        final = PIL.Image.fromarray(np.uint8(img_final))
        final = final.convert("RGBA")
        img2 = PIL.Image.fromarray(np.uint8(original_soil_full_patch))
        img2.paste(final, (0, 0), final)
        final_patch = np.asarray(img2)
        final_image[min_x:max_x, min_y:max_y] = final_patch

    # blur image
    if smoothing == "blur":
        synth_image = cv2.blur(final_image, (3, 3))

    # superpose original edges
    elif smoothing == "original":

        contours, hier = cv2.findContours(mask_, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        edges = np.zeros_like(mask_)

        for c in contours:
            cv2.drawContours(edges, [c], 0, 255, 3)

        edge_img = np.dstack([img, edges])
        edge_img = PIL.Image.fromarray(np.uint8(edge_img))
        edge_img = edge_img.convert("RGBA")
        img2 = PIL.Image.fromarray(np.uint8(final_image))

        img2.paste(edge_img, (0, 0), edge_img)
        img2 = np.asarray(img2)
        synth_image = cv2.blur(img2, (3, 3))

    synth_image = cv2.cvtColor(synth_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{out_dir}/IMG_1171_{i}.png", synth_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])










img_ = np.uint8(img_)
plt.imshow(img_)





# add black pixels to new_soil
idx = np.where(original_soil == (0, 0, 0))
final = np.zeros((new_soil.shape[0], new_soil.shape[1] + int(len(idx[0])/new_soil.shape[0]), 3))
final = PIL.Image.fromarray(np.uint8(final))

img2 = PIL.Image.fromarray(np.uint8(new_soil))
img2 = img2.convert("RGBA")

final.paste(img2, (0, 0), img2)
final = np.asarray(final)
final = final[:, :, :3]
plt.imshow(final)





idx = np.where(subimg == (0, 0, 0))
final = np.zeros((soil.shape[0], soil.shape[1] + int(len(idx[0])/soil.shape[0]), 3))
final = PIL.Image.fromarray(np.uint8(final))
final = final.convert("RGBA")

img2 = PIL.Image.fromarray(np.uint8(soil))
img2 = img2.convert("RGBA")

final.paste(img2, (0, 0), img2)
final = np.asarray(final)
final = final[:, :, :3]
plt.imshow(final)

matched = exposure.match_histograms(img_, final, multichannel=True)
img_ = np.uint8(matched)
plt.imshow(img_)


similar_img = imageio.imread("Z:/Public/Jonas/003_ESWW/ColorTrends/TrainSoil/large_soil/soil_patches/FPWW0120217_RGB1_20160701_115806_1.png")
simimg = similar_img[:, 100:, : 3]
plt.imshow(simimg)

# img_res = cv2.resize(subimg, (205, 200))
# idx = np.where(img_res == (0, 0, 0))
idx = np.where(subimg == (0, 0, 0))
final = np.zeros((simimg.shape[0], simimg.shape[1] + int(len(idx[0])/simimg.shape[0]), 3))
final = PIL.Image.fromarray(np.uint8(final))
final = final.convert("RGBA")

img2 = PIL.Image.fromarray(np.uint8(simimg))
img2 = img2.convert("RGBA")

final.paste(img2, (0, 0), img2)
final = np.asarray(final)
final = final[:, :, :3]

matched = exposure.match_histograms(final, subimg, multichannel=True)
final_matched = matched[0: simimg.shape[0], 0: simimg.shape[1], : 3]
plt.imshow(final_matched)

# THESE EDGES ARE SHIFTED WITH RESPECT TO THE FULL-DEPTH IMAGE!!
fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
axs[0].imshow(final_matched)
axs[0].set_title('img')
axs[1].imshow(similar_img)
axs[1].set_title('orig_mask')
plt.show(block=True)

matched = cv2.resize(final_matched, (192*2, 204*2))
# plt.imshow(matched)

black_pixels_mask = np.all(subimg == [0, 0, 0], axis=-1)
not_black = np.bitwise_not(black_pixels_mask)
plt.imshow(not_black)

matched = matched[:not_black.shape[0], :not_black.shape[1]]

out = np.stack([not_black, not_black, not_black], axis=2) * matched

img = imageio.imread("Z:/Public/Jonas/003_ESWW/ColorTrends/TrainingSpikes/train_IMG_1171.png")
img0 = img
img[min_x:max_x, min_y:max_y] = out
idx = np.all(img == [0, 0, 0], axis=-1)
idx = np.where(idx)

img1 = img0 * not_black


plt.imshow(img)





hist_target = utils.get_hist(img, normalize=False)
hist_source = utils.get_hist(similar_img, normalize=False)

a = np.random.randint(0,10,(3,2,2)) # RGB of size 2x2
b = np.random.randint(0,2,(2,2))    # Binary mask of size 2x2
c = a*b