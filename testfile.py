
import imageio
import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import cv2

image = imageio.imread("Z:/Public/Jonas/Data/ESWW006/ImagesNadir/patches_annotation/all_annotations/images/20220530_Cam_ESWW0060037_Cnp_1_3.png")
plt.imshow(image)

full_img = imageio.imread("Z:/Public/Jonas/Data/ESWW006/ImagesNadir/2022_05_30/JPEG/20220530_Cam_ESWW0060037_Cnp_1.JPG")
dings = full_img[0:4000, 2440:6440]
dings = dings[1400:2600, 0:1200]

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
axs[0].imshow(image)
axs[0].set_title('img')
axs[1].imshow(dings)
axs[1].set_title('orig_mask')
plt.show(block=True)

# ======================================================================================================================
# ======================================================================================================================

# Scale to 0...1
img_RGB = np.array(image / 255, dtype=np.float32)
img_RGB = np.array(image, dtype=np.float32)

# Calculate vegetation indices: ExR, ExG, TGI
R, G, B = cv2.split(img_RGB)
normalizer = np.array(R + G + B, dtype=np.float32)
# Avoid division by zero
normalizer[normalizer == 0] = 1.
r, g, b = (R, G, B) / normalizer

ExR = np.array(1.4 * r - b, dtype=np.float32)
ExG = np.array(2.0 * g - r - b, dtype=np.float32)

GLI = np.array((2 * g - r - b) / (2 * g + r + b), dtype=np.float32)

plt.imshow(GLI)
