from pathlib import Path
import imageio
import os
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import utils
import copy
from scipy import ndimage as ndi
import glob
from skimage.filters.rank import entropy as Entropy
from skimage import morphology

path = Path("C:/Users/anjonas/Pictures/2022_01_12/F2.8_50mm_2m/2022_01_14")

# Change the current working directory
os.chdir(path)

# ======================================================================================================================

depth_map_detail = imageio.imread("DepthMap_A.jpg")
plt.imshow(depth_map_detail)

deep_img = imageio.imread("2022-01-14 16-38-59 (A,Radius1,Smoothing1).tif")
deep_img = (deep_img/256).astype('uint8')

img = imageio.imread("JPG/BF0A3511.JPG")
img_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

img_stack = glob.glob('JPG/*.JPG')

edge_img = np.zeros(img.shape[:2], dtype=np.uint8)

for image in img_stack[:16]:
    print(image)
    img = imageio.imread(image)
    img_blur = cv2.medianBlur(img, 5)
    edges = cv2.Canny(image=img, threshold1=100, threshold2=210)  # Canny Edge Detection
    edges = edges/255
    # plt.imshow(edge_img)
    edge_img = edge_img + edges

# ======================================================================================================================

# post-process edges
pp = np.zeros_like(edge_img)
idx = np.where(edge_img != 0)
pp[idx] = 255
pp = pp.astype("uint8")

pp_filter = utils.filter_objects_size(pp, 1500, "smaller")
pp_filter_blur = cv2.medianBlur(pp_filter, 7)

test = np.bitwise_not(pp_filter_blur)
lost = test*pp_filter

final_filter = lost*255 + pp_filter_blur

plt.imshow(final_filter)

# ======================================================================================================================

plt.imshow(edge_img)
# THESE EDGES ARE SHIFTED WITH RESPECT TO THE FULL-DEPTH IMAGE!!
img = imageio.imread("JPG/BF0A3511.JPG")
fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
axs[0].imshow(deep_img)
axs[0].set_title('img')
axs[1].imshow(img)
axs[1].set_title('orig_mask')
plt.show(block=True)

# ======================================================================================================================

edges = cv2.Canny(image=deep_img, threshold1=70, threshold2=210)  # Canny Edge Detection

depth_map_detail_ = np.where(depth_map_detail <= 172, 0, 255).astype("uint8")
plt.imshow(depth_map_detail_)

# pützle
dmap = utils.filter_objects_size(depth_map_detail_, 300, "smaller")
dmap_inv = np.bitwise_not(dmap)
dmap_inv = utils.filter_objects_size(dmap_inv, 300, "smaller")
dmap = np.bitwise_not(dmap_inv)
plt.imshow(dmap)

combo = edges * dmap

# pützle
combo_ = utils.filter_objects_size(combo, 25, "smaller")
plt.imshow(combo_)

# combine
edges = np.bitwise_not(combo_*255)
combocombo = edges * dmap
plt.imshow(combocombo)

# erode
kernel = np.ones((5, 5), np.uint8)
dmap_er = cv2.erode(dmap, kernel)
plt.imshow(dmap)

# ======================================================================================================================

pp = np.zeros_like(edge_img)
idx = np.where(edge_img != 0)
pp[idx] = 255
pp = pp.astype("uint8")

pp_filter = utils.filter_objects_size(pp, 1500, "smaller")
pp_filter_blur = cv2.medianBlur(pp_filter, 7)

test = np.bitwise_not(pp_filter_blur)
lost = test*pp_filter

final_filter = lost*255 + pp_filter_blur

plt.imshow(final_filter)

ff_inv = np.bitwise_not(final_filter)
ddd = np.stack([depth_map_detail, depth_map_detail, depth_map_detail], axis=2)

DD = ff_inv[1:5463, 1:8191] * depth_map_detail

plt.imshow(DD)


# Plot
fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
axs[0].imshow(deep_img)
axs[0].set_title('img')
axs[1].imshow(edge_img)
axs[1].set_title('orig_mask')
axs[2].imshow(depth_map_detail)
axs[2].set_title('orig_mask')
plt.show(block=True)


img_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_grey = img_grey/255.0

sobelX = ndi.filters.sobel(img_grey, 1)
sobelY = ndi.filters.sobel(img_grey, 0)
img_sobel = np.abs(sobelX) + np.abs(sobelY)
plt.imshow(img_sobel)

edges = cv2.Canny(image=img, threshold1=75, threshold2=150)  # Canny Edge Detection

plt.imshow(edges)


plt.imshow(edges_pp)



# ======================================================================================================================

img = imageio.imread("2022-01-14 16-38-59 (A,Radius1,Smoothing1).tif")
img = (img/256).astype('uint8')
depth_map = imageio.imread("DepthMap_A_2_1.jpg")
depth_map_detail = imageio.imread("DepthMap_A.jpg")

# reduce size
img = img[500:4000, 3000:6000, :]
depth_map = depth_map[500:4000, 3000:6000]
depth_map_detail = depth_map_detail[500:4000, 3000:6000]

# Plot
fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
axs[0].imshow(img)
axs[0].set_title('img')
axs[1].imshow(depth_map_detail)
axs[1].set_title('orig_mask')
plt.show(block=True)

# threshold mask
idx = np.where(np.logical_and(depth_map > 173, depth_map < 256))
mask = np.zeros(img.shape).astype("uint8")
mask[idx] = 255
mask = mask[:, :, 0]

# size filter
mask_sf = utils.filter_objects_size(mask, 2500, "smaller")

# erode mask
kernel = np.ones((5, 5), np.uint8)
mask_sf_er = cv2.erode(mask, kernel)
mask_sf_er = utils.filter_objects_size(mask_sf_er, 2500, "smaller")
mask_sf_er_blur = cv2.medianBlur(mask_sf_er, 21)
plt.imshow(mask_sf_er_blur)

# DETAILED MASK
# threshold mask
idx = np.where(np.logical_and(depth_map_detail > 173, depth_map_detail < 256))
mask_detail = np.zeros(img.shape).astype("uint8")
mask_detail[idx] = 255
mask_detail = mask_detail[:, :, 0]

# erode mask
kernel = np.ones((9, 9), np.uint8)
mask_det = cv2.erode(mask_detail, kernel)
mask_det = utils.filter_objects_size(mask_det, 2500, "smaller")
# plt.imshow(mask_det)

# skeletonize mask
from skimage.morphology import skeletonize
from skimage.segmentation import watershed
skeleton = skeletonize(mask_sf_er_blur/255, method="lee")
# filter skeleton using detailed mask
skeleton = skeleton * mask_det
idx, idy = np.where(skeleton)
subidx, subidy = idx[0::100], idy[0::100]
sk = np.zeros_like(skeleton)
sk[subidx, subidy] = 1
plt.imshow(sk)
start_x, start_y = np.where(sk == 1)
fg_markers = np.zeros_like(sk)
# fg_markers = np.stack([fg_markers, fg_markers, fg_markers], axis=2)
bg_markers = np.zeros_like(sk)
bg_markers[0::20, 0::20] = 1
bg_markers = bg_markers * np.bitwise_not(mask)
plt.imshow(bg_markers)
for i, point in enumerate(zip(start_x, start_y)):
    fg_markers[point] = i+2
markers = bg_markers + fg_markers
plt.imshow(markers)

# markers = np.stack([markers, markers, markers], axis=2)

labels = watershed(img_grey, markers=markers, watershed_line=True, compactness=0)
plt.imshow(labels)


from skimage.segmentation import flood, flood_fill
from skimage import data, filters, color, morphology
img_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_grey = cv2.medianBlur(img_grey, 3)
# img_sobel = filters.sobel(img_grey)

edges = cv2.Canny(image=img, threshold1=100, threshold2=200)  # Canny Edge Detection
edges = edges * mask
plt.imshow(edges)
ed_x, ed_y = np.where(edges==1)
img_grey[ed_x, ed_y] = 0
# img_grey = img_grey * mask
plt.imshow(img_grey)


img[ed_x, ed_y] = (0, 0, 0)
labels = watershed(img, markers=ws_markers, watershed_line=True, compactness=0)

iter = 0
all_out = np.zeros_like(img_grey)



for s_x, s_y in zip(start_x, start_y):
    print(iter)
    test = flood(img_grey, (s_x, s_y), tolerance=0.1)
    all_out = all_out + test
    iter += 1
plt.imshow(all_out)


img_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_grey = cv2.medianBlur(img_grey, 3)
img_sobel = filters.sobel(img_grey)
test = flood(img_sobel, (2158, 1328), tolerance=0.015)

fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
# Show RGB and segmentation mask
axs[0].imshow(mask_sf_er_blur)
axs[0].set_title('img')
axs[1].imshow(control)
axs[1].set_title('orig_mask')
axs[2].imshow(mask_det)
axs[2].set_title('orig_mask')
plt.show(block=True)






# Plot result
fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
# Show RGB and segmentation mask
axs[0].imshow(img)
axs[0].set_title('img')
axs[1].imshow(depth_map)
axs[1].set_title('orig_mask')
axs[2].imshow(mask_sf)
axs[2].set_title('mask')
plt.show(block=True)

from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries

img_Lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
segments_slic = slic(img_Lab, n_segments=25000, compactness=10, sigma=1, start_label=1)

fig, ax = plt.subplots(1, 1, figsize=(10, 10), sharex=True, sharey=True)
ax.imshow(mark_boundaries(img, segments_slic))
ax.set_title('SLIC')
plt.tight_layout()
plt.show()

res = mask_sf/255 * segments_slic
idx = np.unique(res)
out = np.in1d(segments_slic, idx).reshape(segments_slic.shape)
out = np.stack([out, out, out], axis=2)

cleaned = np.where(out, img, (0, 0, 255))
plt.imshow(cleaned)

fig, ax = plt.subplots(1, 1, figsize=(10, 10), sharex=True, sharey=True)
ax.imshow(mark_boundaries(cleaned.astype("uint8"), segments_slic))
ax.set_title('SLIC')
plt.tight_layout()
plt.show()

# skeletonize mask
from skimage.morphology import skeletonize
from skimage.segmentation import watershed


skeleton = skeletonize(mask_sf/255)
plt.imshow(skeleton)

res = skeleton * segments_slic
idx = np.unique(res)
out = np.in1d(segments_slic, idx).reshape(segments_slic.shape)
out = np.stack([out, out, out], axis=2)
cleaned = np.where(out, img, (0, 0, 0))

fig, ax = plt.subplots(1, 1, figsize=(10, 10), sharex=True, sharey=True)
ax.imshow(mark_boundaries(cleaned.astype("uint8"), segments_slic))
ax.set_title('SLIC')
plt.tight_layout()
plt.show()

img_blur = cv2.medianBlur(img, 5)
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)  # Canny Edge Detection
edges_ = np.stack([edges, edges, edges], axis=2)
kernel = np.ones((3, 3), np.uint8)
edges_ = cv2.dilate(edges_, kernel)
cleaned_ = np.where(edges_ == (255, 255, 255), (0, 0, 0), cleaned)
plt.imshow(cleaned_)

sk = np.stack([skeleton, skeleton, skeleton], axis=2)

cleaned_ = np.where(sk, (255, 0, 0), cleaned_)
plt.imshow(cleaned_)



img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
img_hsv = cv2.medianBlur(img_hsv, 11)

mask = flood(img_grey, (2158, 1328), tolerance=0.008)
img_hsv[mask, 0] = 0.5

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
# Show RGB and segmentation mask
axs[0].imshow(img_hsv)
axs[0].set_title('img')
axs[1].imshow(mask)
axs[1].set_title('orig_mask')
plt.show(block=True)

# distance transform
skeleton_num = skeleton.astype("uint8")*255
distance = ndi.distance_transform_edt(np.bitwise_not(skeleton_num))
plt.imshow(distance)

idx = np.where(skeleton)
mask[idx] = 125
distance[idx] = 1000

plt.imshow(mask)

img_blur = cv2.medianBlur(img, 5).astype("uint8")

edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)  # Canny Edge Detection
plt.imshow(edges)

distance = ndi.distance_transform_edt(np.bitwise_not(edges))
plt.imshow(distance)
labels = watershed(distance, watershed_line=True, compactness=0)

combo = mask_sf * distance
plt.imshow(combo)

idx = np.where(combo == 0.)
combo[idx] = 100000

bin = np.zeros_like(combo)
idx = np.where(combo == 100000)
bin[idx] = 1
plt.imshow(bin)

kernel = np.ones((5, 5), np.uint8)
mask_erode = cv2.erode(bin, kernel)
mask_dilate = cv2.dilate(mask_erode, kernel)

out = bin - mask_dilate

out_dilate = cv2.dilate(out, kernel)
scale_percent = 25  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
resized = cv2.resize(out_dilate, dim, interpolation=cv2.INTER_AREA)
idx = np.where(resized != 0.)
resized[idx] = 1
final = skeletonize(resized)
plt.imshow(final)

img_resize = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

edge_pos = np.where(final == 1)
img_resize[edge_pos] = (255, 0, 0 )

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
# Show RGB and segmentation mask
axs[0].imshow(img_resize)
axs[0].set_title('img')
axs[1].imshow(final)
axs[1].set_title('orig_mask')
plt.show(block=True)



idx = np.where(edges == 255)
mask[idx] = 75
distance[idx] = 1000

plt.imshow(distance)

plt.imshow(out)
edges_size_filtered = utils.filter_objects_size(edges, 15, "smaller")


# from scipy import ndimage as ndi
# filled = ndi.binary_fill_holes(edges)
# plt.imshow(filled)

edges_cleaned = edges * mask_erode

from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.future import graph
from skimage import data, segmentation, color

img_blur = cv2.medianBlur(img, 5).astype("uint8")
segments_slic = slic(img_blur, n_segments=1000, compactness=10, sigma=1, start_label=1)

segments_slic = segments_slic[1000:2000, 1000:2000]
img = img[1000:2000, 1000:2000]

out1 = color.label2rgb(segments_slic, img, kind='avg', bg_label=0)
g = graph.rag_mean_color(img, segments_slic)
labels2 = graph.cut_threshold(segments_slic, g, 29)
out2 = color.label2rgb(labels2, img, kind='avg', bg_label=0)

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
# Show RGB and segmentation mask
axs[0].imshow(img)
axs[0].set_title('img')
axs[1].imshow(labels2)
axs[1].set_title('orig_mask')
plt.show(block=True)



fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True,
                       figsize=(6, 8))
ax[0].imshow(out1)
ax[1].imshow(out2)
for a in ax:
    a.axis('off')
plt.tight_layout()



plt.imshow(segments_slic)

filtered_segs = segments_slic * mask_erode/255
plt.imshow(filtered_segs)

kept_segs = np.unique(filtered_segs)

fig, ax = plt.subplots(1, 1, figsize=(10, 10), sharex=True, sharey=True)
ax.imshow(mark_boundaries(img, segments_slic))
ax.set_title('SLIC')
plt.tight_layout()
plt.show()

# Plot result
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
# Show RGB and segmentation mask
axs[0, 0].imshow(img)
axs[0, 0].set_title('img')
axs[1, 0].imshow(depth_map)
axs[1, 0].set_title('orig_mask')
axs[0, 1].imshow(edges_size_filtered)
axs[0, 1].set_title('mask')
axs[1, 1].imshow(mask)
axs[1, 1].set_title('mask')
plt.show(block=True)


# mask_blur = cv2.medianBlur(mask, 29)

# mask_blur = cv2.medianBlur(mask, 29)
# mask_blur = mask_blur[:, :, 0]
# mask_blur = np.bitwise_not(mask_blur)

mask_filtered = utils.filter_objects_size(mask, 10000, "smaller")

plt.imshow(mask_filtered)


_, contours, _ = cv2.findContours(mask_filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

check_img = copy.copy(img)
for c in contours:
    cv2.drawContours(check_img, [c], 0, (255, 0, 0), 2)

# mask = mask[:, :, 0]

# Plot result
fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
# Show RGB and segmentation mask
axs[0].imshow(check_img)
axs[0].set_title('img')
axs[1].imshow(mask_filtered)
axs[1].set_title('orig_mask')
plt.show(block=True)

# ===

path = "O:/Evaluation/FIP/2013/WW002/RGB/2013-07-23_WW002_037-108/JPG"

# ======

img = imageio.imread("C:/Users/anjonas/Pictures/2022_01_12\F2.8_50mm_2m/2022_01_14/2022-01-14 16-38-59 (A,Radius1,Smoothing1).tif")

image = cv2.convertScaleAbs(img, alpha=(255.0/65535.0))

plt.imshow(image)

edges = cv2.Canny(image=image, threshold1=100, threshold2=200)  # Canny Edge Detection

plt.imshow(edges)

depth_map = imageio.imread("DepthMap_A.jpg")

plt.imshow(depth_map)
