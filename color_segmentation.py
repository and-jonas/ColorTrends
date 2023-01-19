from roi_selector import TrainingPatchSelector
import numpy as np
from pathlib import Path
import pickle
import pandas as pd
import glob
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV  # Number of trees in random forest
from sklearn.model_selection import GridSearchCV  # Create the parameter grid based on the results of random search
import SegmentationFunctions
from sklearn.model_selection import cross_val_score

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import cv2

# def run():
#     dir_to_process = "Z:/Public/Jonas/Data/ESWW006/ImagesNadir/patches_annotation_vegcol"
#     dir_positives = "Z:/Public/Jonas/Data/ESWW006/ImagesNadir/patches_annotation_vegcol/yellow"
#     dir_negatives = "Z:/Public/Jonas/Data/ESWW006/ImagesNadir/patches_annotation_vegcol/yellow"
#     dir_control = "Z:/Public/Jonas/Data/ESWW006/ImagesNadir/patches_annotation_vegcol/checkers1"
#     roi_selector = TrainingPatchSelector(dir_to_process, dir_positives, dir_negatives, dir_control)
#     roi_selector.iterate_images()
#
#
# if __name__ == '__main__':
#     run()
#
#
# # ======================================================================================================================
# # (1) extract features and save to .csv
# # ======================================================================================================================
#
# workdir = 'Z:/Public/Jonas/Data/ESWW006/ImagesNadir/patches_annotation_vegcol'
#
# # set directories to previously selected training patches
# dir1 = f'{workdir}/green'
# dir2 = f'{workdir}/yellow'
# dir3 = f'{workdir}/brown'
#
# # extract feature data for all pixels in all patches
# # output is stores in .csv files in the same directories
# SegmentationFunctions.iterate_patches_dirs2([dir1, dir2, dir3])
#
# # ======================================================================================================================
# # (2) combine training data from all patches into single file
# # ======================================================================================================================
#
# # import all training data
# # get list of files
# files1 = glob.glob(f'{dir1}/*.csv')
# files2 = glob.glob(f'{dir2}/*.csv')
# files3 = glob.glob(f'{dir3}/*.csv')
#
# all_files = files1 + files2 + files3
#
# # load data
# train_data = []
# for i, file in enumerate(all_files):
#     print(i)
#     data = pd.read_csv(file)
#     target = file.split("/")[7].split("\\")[0]
#     data['response'] = target
#     data = data.sample(n=7, axis=0, replace=True)
#     train_data.append(data)
# # to single df
# train_data = pd.concat(train_data)
# # export, this may take a while
# train_data.to_csv(f'{workdir}/training_data.csv', index=False)
#
# # ======================================================================================================================
# # (3) train random forest classifier
# # ======================================================================================================================
#
# workdir = 'Z:/Public/Jonas/Data/ESWW006/ImagesNadir/patches_annotation_vegcol'
#
# train_data = pd.read_csv(f'{workdir}/training_data.csv')
#
# # OPTIONAL: sample an equal number of rows per class for training
# # THIS INCREASES THE CROSS-VALIDATED CLASS-BALANCED ACCURACY FROM 0.943 to 0.970
# n_brown = train_data.groupby('response').count().iloc[0, 0]
# n_green = train_data.groupby('response').count().iloc[1, 0]
# n_yellow = train_data.groupby('response').count().iloc[2, 0]
#
# n_min = min(n_brown, n_green, n_yellow)
# train_data = train_data.groupby(['response']).apply(lambda grp: grp.sample(n=n_min))
#
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
#
# # predictor matrix
# X = np.asarray(train_data)[:, 0:21]
# # response vector
# y = np.asarray(train_data)[:, 21]
#
# model = rf_random.fit(X, y)
# rf_random.best_params_
# best_random = rf_random.best_estimator_
#
# # result is {'n_estimators': 160, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': 20, 'bootstrap': False}
#
# param_grid = {
#     'bootstrap': [False],
#     'max_depth': [16, 18, 20, 22, 24],
#     'max_features': ['auto'],
#     'min_samples_leaf': [1, 2, 3, 4],
#     'min_samples_split': [3, 5, 7, 9],
#     'n_estimators': [150, 160, 170]
# }
# # Create a based model
# rf = RandomForestClassifier()  # Instantiate the grid search model
# grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
#                            cv=10, n_jobs=-1, verbose=3)
#
# # Fit the grid search to the data
# grid_search.fit(X, y)
# grid_search.best_params_
#
# # specify model hyper-parameters
# clf = RandomForestClassifier(
#     max_depth=20,  # maximum depth of 95 decision nodes for each decision tree
#     max_features='auto',  # maximum of 6 features (channels) are considered when forming a decision node
#     min_samples_leaf=2,  # minimum of 6 samples needed to form a final leaf
#     min_samples_split=5,  # minimum 4 samples needed to create a decision node
#     n_estimators=160,  # maximum of 55 decision trees
#     bootstrap=False,  # don't reuse samples
#     random_state=1,
#     n_jobs=-1
# )
#
# # fit random forest
# model = clf.fit(X, y)
# score = model.score(X, y)  # not cross-validated
# scores = cross_val_score(clf, X, y, cv=10, scoring='balanced_accuracy')
# score = scores.mean()
#
# # save model
# path = f'{workdir}/Output/Models'
# if not Path(path).exists():
#     Path(path).mkdir(parents=True, exist_ok=True)
# pkl_filename = f'{path}/rf_colveg_segmentation.pkl'
# with open(pkl_filename, 'wb') as file:
#     pickle.dump(model, file)

# ======================================================================================================================


from ImagePreSegmentor import ImagePostSegmentor
# from ImagePreSegmentorSerial import ImagePostSegmentor


import os

workdir = 'Z:/Public/Jonas/Data/ESWW006/ImagesNadir'

dirs = [f for f in os.listdir(workdir)]
# dirs = [os.path.join(workdir, d) for d in dirs if d  not in ['2022_06_01', '2022_06_23', '2022_05_28']]
dirs = [os.path.join(workdir, d) for d in dirs]
dirs = [d + "/JPEG" for d in dirs]


def run():
    dirs_to_process = dirs
    base_dir = f'{workdir}/Output'
    dir_patch_coordinates = f'{workdir}/Meta/patch_coordinates'
    dir_veg_masks = f'{workdir}/Output/SegVeg/Mask'
    dir_ear_masks = f'{workdir}/Output/SegEar/Mask'
    dir_output = f'{workdir}/Output/ColSeg'
    dir_model = f'{workdir}/patches_annotation_vegcol/Output/Models/rf_colveg_segmentation.pkl'
    image_post_segmentor = ImagePostSegmentor(
        base_dir=base_dir,
        dirs_to_process=dirs_to_process,
        dir_patch_coordinates=dir_patch_coordinates,
        dir_veg_masks=dir_veg_masks,
        dir_ear_masks=dir_ear_masks,
        dir_model=dir_model,
        dir_output=dir_output,
        img_type="JPG",
        mask_type="png",
        overwrite=True,
        save_masked_images=False
    )
    image_post_segmentor.process_images()


if __name__ == "__main__":
    run()

