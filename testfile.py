
import imageio

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

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