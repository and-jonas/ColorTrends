import imageio
import numpy as np
import cv2

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from plantcv import plantcv as pcv


# ======================================================================================================================



# ======================================================================================================================

img = imageio.imread("Z:/Public/Jonas/003_ESWW/ColorTrends/TrainingSegmentation/2013-07-16\IMG_0996.JPG")

# mask for paper background
lower_white = np.array([245, 245, 245], dtype=np.uint8)  # threshold for white pixels
upper_white = np.array([255, 255, 255], dtype=np.uint8)
mask1 = cv2.inRange(img, lower_white, upper_white)  # could also use threshold

# # Plot result
# fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
# # Show RGB and segmentation mask
# axs[0].imshow(mask1)
# axs[0].set_title('img')
# axs[1].imshow(mask_blur)
# axs[1].set_title('orig_mask')
# plt.show(block=True)

# get contours
_, contours, _ = cv2.findContours(mask_blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img_ = img.copy()
cimg = np.zeros(img.shape)

rectangles = []
for c in contours:
   rect = cv2.minAreaRect(c)
   (x, y), (w, h), a = rect  # a - angle
   box = cv2.boxPoints(rect)
   box = np.int0(box)  # turn into ints
   # rect2 = cv2.drawContours(img_, [box], 0, (0, 0, 255), 5)
   rectangles.append(rect)

x_white = np.max([item[0][0] for item in rectangles])
y_white = np.max([item[0][1] for item in rectangles])
x_diff = rectangles[0][0][0] - rectangles[1][0][0]
y_diff = rectangles[0][0][1] - rectangles[1][0][1]

xs = []
ys = []

for c in range(6):
   x = x_white - c * x_diff
   y = y_white - c * y_diff
   xs.append(x)
   ys.append(y)
   # draw
   rect = (x, y), (w, h), a
   box = cv2.boxPoints(rect)
   box = np.int0(box) #turn into ints
   rect2 = cv2.drawContours(img_,[box],0,(0, 0, 255), 5)
   cimg = cv2.drawContours(cimg,[box],0, (255, 255, 255), -1)

# plt.imshow(rect2)

out_x = []
out_y = []
for i in range(4):
   x = []
   y = []
   for j in range(6):
      x__ = xs[j] - ((i+1)*y_diff)
      y__ = ys[j] + (i*x_diff + 1.1*w)
      x.append(x__)
      y.append(y__)
   out_x.append(x)
   out_y.append(y)

for x,y in zip(out_x, out_y):
   print(x)
   print(y)
   for i in range(6):
      rect = (x[i], y[i]), (w, h), a
      print(rect)
      box = cv2.boxPoints(rect)
      box = np.int0(box) #turn into ints
      rect2 = cv2.drawContours(img_, [box], 0, (0, 0, 255), 5)
      cimg = cv2.drawContours(cimg, [box], 0, (255, 255, 255), -1)
plt.imshow(cimg)

kernel = np.ones((25, 25), np.uint8)
mask = cv2.erode(cimg, kernel)
mask = mask.astype(dtype=np.uint8)

checker = cv2.bitwise_and(img, mask)
plt.imshow(checker)

mask_ = np.ascontiguousarray(mask[:, :, 0])

n_comps, output, stats, centroids = cv2.connectedComponentsWithStats(mask_, connectivity=8)

# ======================================================================================================================
# get average color
# ======================================================================================================================

_, contours, _ = cv2.findContours(mask_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize empty list
lst_intensities = []

# For each list of contour points...
medians = []
for i in range(len(contours)):
    # Create a mask image that contains the contour filled in
    cimg = np.zeros_like(mask_)
    cv2.drawContours(cimg, contours, i, color=255, thickness=-1)

    # Access the image pixels and create a 1D numpy array then add to list
    pts = np.where(cimg == 255)
    pixels = img[pts[0], pts[1]]
    med = np.median(pixels, axis=0).astype("int")
    medians.append(med)






