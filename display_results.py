
import numpy as np
import imageio
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
from pathlib import Path

dir = "C:/Users/anjonas/PycharmProjects/DL/data/inference_test"
images = glob.glob(f'{dir}/*.png')

for image in images:

    img_name = os.path.basename(image)
    stem_name = Path(image).stem

    mask = imageio.imread(f'{dir}/predictions/{stem_name}_mask.png')
    img = imageio.imread(f'{dir}/{img_name}')
    # plt.imshow(mask)
    # plt.imshow(img)

    mask = mask/255.0

    M = mask.ravel()
    M = np.expand_dims(M, -1)

    outmask = np.dot(M, np.array([[0, 0, 255, 90]]))
    outmask = np.reshape(outmask, newshape=(1956, 1956, 4))
    outmask = outmask.astype("uint8")

    # plt.imshow(outmask)

    mask = Image.fromarray(outmask, mode="RGBA")
    img = Image.fromarray(img, mode="RGB")
    img = img.convert("RGBA")
    img.paste(mask, (0, 0), mask)
    final_patch = np.asarray(img)
    plt.imshow(final_patch)
    imageio.imwrite(f'{dir}/overlay/{stem_name}_overlay.png', final_patch)
