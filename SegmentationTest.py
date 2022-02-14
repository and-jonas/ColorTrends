
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import imageio
import os


path = "Z:/Public/Jonas/003_ESWW/ColorTrends/Training/IMG_1171.png"
img = imageio.imread(path)
plt.imshow(img)

# name = path.split("/")
# img_name = name[6] + "_" + name[8]
img_name = os.path.basename(path)

img_ = img[250:(250+2048), 1200:(1200+2048)]
plt.imshow(img_)

imageio.imsave(f"Z:/Public/Jonas/003_ESWW/ColorTrends/Training/train_{img_name}", img_)

