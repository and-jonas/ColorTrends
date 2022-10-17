import glob
import os.path

import imageio
import numpy as np

import matplotlib as mpl
import pandas as pd

mpl.use('Qt5Agg')
import matplotlib.pyplot as plt

from plantcv import plantcv as pcv

from scipy import ndimage
import cv2
import csv

out_dir = "Z:/Public/Jonas/Data/ESWW006/ImagesNadir/patches_analysis"
in_dir = "Z:/Public/Jonas/Data/ESWW006/ImagesNadir/2022_*/JPEG"
images = glob.glob(f'{in_dir}/*.JPG')
images = [item for item in images if "Ref" not in item]

# color checker should be more or less centered. Detect color checker and use central coordinates for patch selection.
with open("Z:/Public/Jonas/Data/ESWW006/ImagesNadir/Meta/patch_coordinates.csv", 'w', newline='') as f1:
    writer = csv.writer(f1, delimiter=',')
    for i in images:
        base_name = os.path.basename(i)
        png_name = base_name.replace(".JPG", ".png")
        img = imageio.imread(i)
        try:
            dataframe1, start, space = pcv.transform.find_color_card(rgb_img=img, background='light')
        except:
            dataframe1, start, space = pcv.transform.find_color_card(rgb_img=img, background='dark')

        source_mask = pcv.transform.create_color_card_mask(img, radius=10, start_coord=start,
                                                           spacing=space, nrows=4, ncols=6)

        binary = np.where(source_mask == 0, 0, 1)
        center = ndimage.center_of_mass(binary)
        x1 = int(center[1]-2250)
        x2 = int(center[1]+2250)
        y1 = 0
        y2 = 4500
        coords = list((base_name, x1, x2, y1, y2))
        row = coords
        writer.writerow(row)

    # df = pd.DataFrame([coordinates])
    # df2 = df.set_axis(['x1', 'x2', 'y1', 'y2'], axis=1, inplace=False)
    # csv_namer = str(corrected_patch_name).replace(".png", ".csv")
    # df2.to_csv(csv_namer, index=False)
    #
    # x1, x2, y1, y2 = coordinates
    # imageio.imwrite(original_patch_name, soil_patch)
    # coordinates = list(coordinates)
    # df = pd.DataFrame([coordinates])


