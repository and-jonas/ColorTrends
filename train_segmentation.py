
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
from importlib import reload
reload(SegmentationFunctions)
import imageio
import cv2
import random
import os

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from skimage import morphology
import math
import copy
from skimage.filters.rank import entropy as Entropy
from scipy import ndimage

# ======================================================================================================================
# (1) extract features and save to .csv
# ======================================================================================================================

workdir = 'Z:/Public/Jonas/003_ESWW/ColorTrends/TrainingSegmentation'

dir_names = f'{workdir}/ImageSets/Segmentation'

with open(f'{dir_names}/default.txt') as f:
    image_names = f.read().splitlines()

dir_images = f'{workdir}/AllDates'
dir_masks = f'{workdir}/SegmentationClass'

for image_name in image_names:

    full_name = f'{dir_images}/{image_name}.JPG'
    full_name_mask = f'{dir_masks}/{image_name}.png'
    img = imageio.imread(full_name)
    mask = imageio.imread(full_name_mask)

    ## inserted
    # convert pixels to gray scale
    graypatch = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # get entropy as texture measure
    graypatch_sm = cv2.medianBlur(graypatch, 31)
    img_ent = Entropy(graypatch_sm, morphology.disk(10))

    # Get x-gradient in "sx"
    sx = ndimage.sobel(graypatch, axis=0, mode='constant')
    # Get y-gradient in "sy"
    sy = ndimage.sobel(graypatch, axis=1, mode='constant')
    # Get square root of sum of squares
    sobel = np.hypot(sx, sy)

    # Hopefully see some edges
    plt.imshow(sobel, cmap=plt.cm.gray)
    plt.show()


    ### end inserted

    newmask = np.zeros(mask.shape[:2])
    newmask[np.where(mask == 210)[:2]] = 255
    newmask = newmask.astype("uint8")
    df1 = SegmentationFunctions.iterate_patches(image=img, mask=newmask)
    df1['response'] = 'pos'

    newmask = np.zeros(mask.shape[:2])
    newmask[np.where(mask == 169)[:2]] = 255
    newmask = newmask.astype("uint8")
    df2 = SegmentationFunctions.iterate_patches(image=img, mask=newmask)
    df2['response'] = 'neg'

    result = pd.concat([df1, df2])

    result.to_csv(f'{workdir}/TrainingData/{image_name}_data.csv', index=False)


# ======================================================================================================================
# (2) combine training data from all patches into single file
# ======================================================================================================================

# import all training data
# get list of files
files = glob.glob(f'{workdir}/TrainingData/*.csv')

# load data
train_data = []
for i, file in enumerate(files):
    print(i)
    data = pd.read_csv(file)
    train_data.append(data)
# to single df
train_data = pd.concat(train_data)
# export, this may take a while
train_data.to_csv(f'{workdir}/TrainingData/training_data.csv', index=False)

# ======================================================================================================================
# (3) train random forest classifier
# ======================================================================================================================

train_data = pd.read_csv(f'{workdir}/TrainingData/training_data.csv')

# OPTIONAL: sample an equal number of rows per class for training
n_pos = train_data.groupby('response').count().iloc[0, 0]
n_neg = train_data.groupby('response').count().iloc[1, 0]
n_min = min(n_pos, n_neg)
train_data = train_data.groupby(['response']).apply(lambda grp: grp.sample(n=n_min))

# n_estimators = [int(x) for x in np.linspace(start=20, stop=200, num=10)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4, 8]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]  # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
#
#
# # Use the random grid to search for best hyperparameters
# # First create the base model to tune
# rf = RandomForestClassifier()
# # Random search of parameters, using 3 fold cross validation,
# # search across 100 different combinations, and use all available cores
# rf_random = RandomizedSearchCV(estimator=rf,
#                                param_distributions=random_grid,
#                                n_iter=100, cv=10,
#                                verbose=3, random_state=42,
#                                n_jobs=-1)  # Fit the random search model

# predictor matrix
X = np.asarray(train_data)[:, 0:21]
# response vector
y = np.asarray(train_data)[:, 21]

# model = rf_random.fit(X, y)
# rf_random.best_params_
# best_random = rf_random.best_estimator_
#
# param_grid = {
#     'bootstrap': [True],
#     'max_depth': [60, 70, 80],
#     'max_features': [2, 4, 6, 8, 10],
#     'min_samples_leaf': [2, 4, 6, 8],
#     'min_samples_split': [8, 10, 12],
#     'n_estimators': [200, 300, 400]
# }
# # Create a based model
# rf = RandomForestClassifier()  # Instantiate the grid search model
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
#                            cv=10, n_jobs=-1, verbose=3)
#
# # Fit the grid search to the data
# grid_search.fit(X, y)
# grid_search.best_params_

# specify model hyper-parameters
clf = RandomForestClassifier(
    max_depth=80,  # maximum depth of 95 decision nodes for each decision tree
    max_features=2,  # maximum of 6 features (channels) are considered when forming a decision node
    min_samples_leaf=8,  # minimum of 6 samples needed to form a final leaf
    min_samples_split=12,  # minimum 4 samples needed to create a decision node
    n_estimators=200,  # maximum of 55 decision trees
    bootstrap=False,  # don't reuse samples
    random_state=1,
    n_jobs=-1
)

# fit random forest
model = clf.fit(X, y)
score = model.score(X, y)

# save model
path = f'{workdir}/Output/Models'
if not Path(path).exists():
    Path(path).mkdir(parents=True, exist_ok=True)
pkl_filename = f'{path}/rf_segmentation.pkl'
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)

# ======================================================================================================================
# predict image
# ======================================================================================================================

dir_model = f'{workdir}/Output/Models/rf_segmentation.pkl'
mask = SegmentationFunctions.segment_image(image=img, dir_model=dir_model)
mask = np.bitwise_not(mask)
imageio.imsave(f'{workdir}/Output/masks/{image_name}.png', mask)




mask_pp = cv2.medianBlur(mask, 25).astype("uint8")
mask_pp = np.dstack([mask_pp, mask_pp, mask_pp])
img_pp = cv2.bitwise_and(img, mask_pp)

edges = cv2.Canny(graypatch, 100, 200)

result = filter_objects_size(edges, 200, "smaller")


result = [idx for idx, element in enumerate(sizes) if element > 50]

out = np.in1d(output, result).reshape(output.shape)

plt.imshow(result)

def filter_objects_size(mask, size_th, dir):
    """
    Filter objects in a binary mask by size
    :param mask: A binary mask to filter
    :param size_th: The size threshold used to filter (objects GREATER than the threshold will be kept)
    :return: A binary mask containing only objects greater than the specified threshold
    """
    _, output, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sizes = stats[1:, -1]
    if dir == "greater":
        idx = (np.where(sizes > size_th)[0] + 1).tolist()
    if dir == "smaller":
        idx = (np.where(sizes < size_th)[0] + 1).tolist()
    out = np.in1d(output, idx).reshape(output.shape)
    cleaned = np.where(out, 0, mask)

    return cleaned


cleaned = np.where(out, 0, mask)
cleaned = np.uint8(np.where(output == index + 1, 1, 0))



# Hopefully see some edges
# # Plot result
fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
# Show RGB and segmentation mask
axs[0, 0].imshow(img)
axs[0, 0].set_title('orig_mask')
axs[1, 0].imshow(mask)
axs[1, 0].set_title('orig_mask')
axs[0, 1].imshow(edges)
axs[0, 1].set_title('orig_mask')
axs[1, 1].imshow(mask_pp)
axs[1, 1].set_title('orig_mask')
plt.show(block=True)



# # Plot result
fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
# Show RGB and segmentation mask
axs[0].imshow(mask)
axs[0].set_title('img')
axs[1].imshow(img)
axs[1].set_title('orig_mask')
axs[2].imshow(img_pp)
axs[2].set_title('orig_mask')
plt.show(block=True)
