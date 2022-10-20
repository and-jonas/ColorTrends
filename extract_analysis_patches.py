import glob
import os.path

import imageio
import numpy as np

import pandas as pd

from plantcv import plantcv as pcv
from scipy import ndimage
import cv2
import copy

out_dir = "Z:/Public/Jonas/Data/ESWW006/ImagesNadir"
in_dir = "Z:/Public/Jonas/Data/ESWW006/ImagesNadir/2022_*/JPEG"
images = glob.glob(f'{in_dir}/*.JPG')
images = [item for item in images if "Ref" not in item]
images = images[468:]

# color checker should be more or less centered. Detect color checker and use central coordinates for patch selection.
for i in images:
    base_name = os.path.basename(i)
    stem_name = base_name.replace(".JPG", "")
    png_name = base_name.replace(".JPG", ".png")
    img = imageio.imread(i)
    try:
        dataframe1, start, space = pcv.transform.find_color_card(rgb_img=img, background='light')
    except:
        dataframe1, start, space = pcv.transform.find_color_card(rgb_img=img, background='dark')

    try:
        source_mask = pcv.transform.create_color_card_mask(img, radius=10, start_coord=start,
                                                           spacing=space, nrows=4, ncols=6)
    except RuntimeError:
        print(f'failed for {base_name}')
        source_mask = None

    if source_mask is not None:
        # get corner coordinates and write to csv
        binary = np.where(source_mask == 0, 0, 1)
        center = ndimage.center_of_mass(binary)
        x1 = int(center[1]-2250) # 4000PX X 4000PX patches
        x2 = int(center[1]+1750)
        y1 = 0
        y2 = 4000
    else:
        x1 = 2750
        x2 = 6750
        y1 = 0
        y2 = 4000
    coordinates = list((x1, x2, y1, y2))
    df = pd.DataFrame([coordinates])
    df2 = df.set_axis(['x1', 'x2', 'y1', 'y2'], axis=1, inplace=False)
    df2.to_csv(f'{out_dir}/Meta/patch_coordinates/{stem_name}.txt', index=False)

    # create checker image
    checker = copy.copy(img)
    cv2.rectangle(checker, (x1, y1), (x2, y2), (255, 0, 0), 9)
    x_new = int(checker.shape[0]/10)
    y_new = int(checker.shape[1]/10)
    checker = cv2.resize(checker, (y_new, x_new), interpolation=cv2.INTER_LINEAR)
    checker_name = f'{out_dir}/Meta/patch_checkers/{base_name}'
    imageio.imwrite(checker_name, checker)
