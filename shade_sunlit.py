
# ======================================================================================================================
# Author: Jonas Anderegg, jonas.anderegg@usys.ethz.ch
# Extract training data from labelled patches and train segmentation algorithm
# Last modified: 2021-11-10
# ======================================================================================================================

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV  # Number of trees in random forest
from sklearn.model_selection import GridSearchCV  # Create the parameter grid based on the results of random search
import SegmentationFunctions

import imageio
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

# ======================================================================================================================
# (1) extract features and save to .csv
# ======================================================================================================================

workdir = 'Z:/Public/Jonas/003_ESWW/ColorTrends'

# set directories to previously selected training patches
dir_positives = f'{workdir}/shade_sunlit/sunlit'
dir_negatives = f'{workdir}/shade_sunlit/shade'

# extract feature data for all pixels in all patches
# output is stores in .csv files in the same directories
SegmentationFunctions.iterate_patches_dirs(dir_positives, dir_negatives)

# ======================================================================================================================
# (2) combine training data from all patches into single file
# ======================================================================================================================

# import all training data
# get list of files
files_pos = glob.glob(f'{dir_positives}/*.csv')
files_neg = glob.glob(f'{dir_negatives}/*.csv')
all_files = files_pos + files_neg

# load data
train_data = []
for i, file in enumerate(all_files):
    print(i)
    data = pd.read_csv(file)
    # data = data.iloc[::10, :]  # only keep every 10th pixel of the patch
    data_mean = data.mean().to_frame().T
    data_mean['response'] = data['response'][0]
    train_data.append(data_mean)
# to single df
train_data = pd.concat(train_data)
# export, this may take a while
train_data.to_csv(f'{workdir}/shade_sunlit/training_data.csv', index=False)

# ======================================================================================================================

testimg = imageio.imread('Z:/Public/Jonas/003_ESWW/ColorTrends/shade_sunlit/test_img/IMG_0285.JPG')

plt.imshow(testimg)