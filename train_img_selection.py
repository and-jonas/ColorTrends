
# ======================================================================================================================
# Author: Jonas Anderegg, jonas.anderegg@usys.ethz.ch
# Project: ColorTrends
# Get image light contrast and image exif data for training data set selection
# ======================================================================================================================

import glob
import imageio
import numpy as np
import pandas as pd
import utils
from sklearn.cluster import KMeans
import exifread
import os

base_dir = "O:/Evaluation/FIP/2013/WW002/RGB"

# ======================================================================================================================

# measurement dates to use
dirs = os.listdir(base_dir)
dirs = dirs[2:12]
dirs = [k for k in dirs if '-108' in k]

# get full directory names
full_dirs = []
for dir in dirs:
    full_dir = f'{base_dir}/{dir}/JPG'
    full_dirs.append(full_dir)

# extract intensity histograms
dfs = []
for dir in full_dirs:

    # list all files
    files = glob.glob(f'{dir}/*.JPG')

    # get r, g, b histograms
    all_hists = []
    base_names = []
    for file in files:
        img = imageio.imread(file)
        base_name = os.path.basename(file)

        # rotate image if not landscape
        if img.shape[0] > img.shape[1]:
            img = np.rot90(img)

        hists = utils.get_hist(img)[0]
        all_hists.append(hists)
        base_names.append(base_name)

    # to data frame
    df = pd.DataFrame(all_hists)
    df = df.set_axis(base_names, axis=0, inplace=False)
    dfs.append(df)

out_df = pd.concat(dfs, axis=0)

out_df.to_csv("Z:/Public/Jonas/003_ESWW/ColorTrends/train_img_selection/rgb_histograms.csv")

# cluster images based on r, g, b intensity histograms
k_means = KMeans(n_clusters=3, random_state=0).fit(out_df)
y = k_means.fit_predict(out_df)
centroids = k_means.cluster_centers_
out_df['Cluster'] = y
cents = pd.DataFrame(centroids)
cents['Cluster'] = list(range(3))

out_df.reset_index(inplace=True)

to_disk = out_df[["index", "Cluster"]]

# ======================================================================================================================

# get image exif data
dfs = []
for dir in full_dirs:
    print(dir)
    # list all files
    files = glob.glob(f'{dir}/*.JPG')

    # get image exif data
    img_id = []
    datetime = []
    for file in files:
        print(file)
        with open(file, 'rb') as f:
            tags = exifread.process_file(f)
        v = tags['Image DateTime'].values
        name = os.path.basename(file)
        img_id.append(name)
        datetime.append(v)
    d = {'img_id': img_id, 'datetime': datetime}
    df = pd.DataFrame(d)
    dfs.append(df)

out_df = pd.concat(dfs, axis=0)

out_df.to_csv("Z:/Public/Jonas/003_ESWW/ColorTrends/train_img_selection/exifdata.csv", index=False)





