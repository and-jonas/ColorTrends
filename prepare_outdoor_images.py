
import glob
import imageio
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from pathlib import Path
import copy
import shutil

# ======================================================================================================================
# TEST
# ======================================================================================================================

# path = "Z:/Public/Jonas/003_ESWW/ColorTrends/testing_segmentation_outdoor/2022_04_19/JPEG"
#
# image_paths= glob.glob(f"{path}/*.JPG")[:18]
# for image_path in image_paths:
#     basename = os.path.basename(image_path)
#     img = imageio.imread(image_path)
#     img_ = img[1157:2224, 3688:5212]
#     imageio.imwrite(f'{path}/cropped/{basename}', img_)
#
# image_paths = glob.glob(f"{path}/*.JPG")[18:]
# for image_path in image_paths:
#     basename = os.path.basename(image_path)
#     img = imageio.imread(image_path)
#     img_ = img[1715:2782, 3688:5212]
#     # plt.imshow(img_)
#     imageio.imwrite(f'{path}/cropped/{basename}', img_)
#
#
# # ======================================================================================================================
# # ======================================================================================================================
#
# # ======================================================================================================================
# REAL
# (1) Extract image patches with blue background
# # ======================================================================================================================
#
# paths = glob.glob("Z:/Public/Jonas/Data/ESWW006/Images_trainset/*/JPEG/*.JPG")
# parent_dir = Path("Z:/Public/Jonas/Data/ESWW006/Images_trainset")
# meas_events = [x for x in os.listdir(parent_dir) if x.startswith("2022_")]
#
# for event in meas_events:
#
#     dir_event = parent_dir / event
#     output_dir = dir_event / "Processed"
#     checker_dir = output_dir / "CheckImages"
#     patch_dir = output_dir / "Patches"
#     output_dir.mkdir(parents=True, exist_ok=True)
#     checker_dir.mkdir(parents=True, exist_ok=True)
#     patch_dir.mkdir(parents=True, exist_ok=True)
#
#     # check if already processed
#     if not os.listdir(checker_dir):
#         pass
#     else:
#         print("already processed - skipping")
#         continue
#
#     paths = glob.glob(f'{dir_event}/JPEG/ESWW*.JPG')  # rejects the soil images
#
#     for p in paths:
#
#         image_name = os.path.basename(p)
#         image_png_name = image_name.replace(".JPG", ".png")
#
#         print(image_name)
#
#         # find blue areas in image
#         image = cv2.imread(p)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#         image_cut = image[:4500, :, :]  # remove the color checker
#         hsv = cv2.cvtColor(image_cut, cv2.COLOR_RGB2HSV)
#         lower = np.array([95, 150, 40])
#         upper = np.array([115, 200, 255])
#         mask = cv2.inRange(hsv, lower, upper)
#         mask = cv2.medianBlur(mask, 13)
#
#         # Find contours
#         cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         cnts = np.concatenate(cnts)
#
#         # get minimum area bounding rectangle and shrink
#         rect = cv2.minAreaRect(cnts)
#         sizes = list(rect[1])
#         sizes = [s - 200 for s in sizes]
#         rect_ = (rect[0], (sizes[0], sizes[1]), rect[2])
#         box = cv2.boxPoints(rect_)
#         box = np.int0(box)
#
#         # create a check image
#         checker_img = copy.copy(image)
#         checker_img = cv2.drawContours(checker_img, [box], 0, (255, 0, 0), 5)
#         x_new = int(checker_img.shape[0]/4)
#         y_new = int(checker_img.shape[1]/4)
#         checker_img = cv2.resize(checker_img, (y_new, x_new), interpolation=cv2.INTER_LINEAR)
#
#         # cut and rotate
#         rows, cols = image.shape[0], image.shape[1]
#         if rect_[2] >= 45:
#             M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rect_[2]-90, 1)
#         else:
#             M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rect_[2], 1)
#         img_rot = cv2.warpAffine(image, M, (cols, rows))
#         checkimg_rot = cv2.warpAffine(checker_img, M, (cols, rows))
#
#         # rotate bounding box
#         rect0 = (rect_[0], rect_[1], 0.0)
#         pts = np.int0(cv2.transform(np.array([box]), M))[0]
#         pts[pts < 0] = 0
#
#         # crop
#         if rect_[2] >= 45:
#             img_crop = img_rot[pts[1][1]:pts[2][1],
#                        pts[0][0]:pts[1][0]]
#         else:
#             img_crop = img_rot[pts[1][1]:pts[0][1],
#                        pts[1][0]:pts[2][0]]
#
#         # save output
#         checker_name = checker_dir / image_name
#         patch_name = patch_dir / image_png_name
#         imageio.imwrite(checker_name, checker_img)
#         imageio.imwrite(patch_name, img_crop)
#
# # ======================================================================================================================
# # (2) copy training patches from each date into a single directory for further processing
# # ======================================================================================================================
#
# patches = glob.glob(f'{parent_dir}/2022*/Processed/Patches/*.png')
#
# target_dir = parent_dir / "Patches"
#
# for patch in patches:
#     path = os.path.normpath(patch)
#     components = path.split(os.sep)
#     date = [i for i in components if i.startswith('2022_')][0].replace("_", "")
#     file_name_in = os.path.basename(patch)
#     plot = file_name_in.replace(".png", "")
#     patch_id = plot + "_" + date
#     file_name_out = patch_id + ".png"
#     full_file_name_out = target_dir / file_name_out
#     if not os.path.exists(full_file_name_out):
#         shutil.copy(patch, full_file_name_out)
#
# ======================================================================================================================
# # Author: Jonas Anderegg, jonas.anderegg@usys.ethz.ch
# # Project: LesionZoo
# # Date: 15.03.2021
# # Sample training data blue background removal
# # ======================================================================================================================
#
#
# from roi_selector import TrainingPatchSelector
#
#
# def run():
#     dir_to_process = "Z:/Public/Jonas/Data/ESWW006/Images_trainset/Patches"
#     dir_positives = "Z:/Public/Jonas/Data/ESWW006/Images_trainset/foreground"
#     dir_negatives = "Z:/Public/Jonas/Data/ESWW006/Images_trainset/background"
#     dir_control = "Z:/Public/Jonas/Data/ESWW006/Images_trainset/checkers"
#     roi_selector = TrainingPatchSelector(dir_to_process, dir_positives, dir_negatives, dir_control)
#     roi_selector.iterate_images()
#
#
# if __name__ == '__main__':
#     run()
#
# # ======================================================================================================================
#
# # ======================================================================================================================
# # Author: Jonas Anderegg, jonas.anderegg@usys.ethz.ch
# # Extract training data from labelled patches and train segmentation algorithm
# # Last modified: 2021-11-10
# # ======================================================================================================================
#
# from pathlib import Path
# import pickle
# import numpy as np
# import pandas as pd
# import glob
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import RandomizedSearchCV  # Number of trees in random forest
# from sklearn.model_selection import GridSearchCV  # Create the parameter grid based on the results of random search
# import SegmentationFunctions
#
# import imageio
# import matplotlib
# matplotlib.use('Qt5Agg')
# import matplotlib.pyplot as plt
#
# import cv2
#
# # ======================================================================================================================
# # (1) extract features and save to .csv
# # ======================================================================================================================
#
# workdir = 'Z:/Public/Jonas/Data/ESWW006/Images_trainset'
#
# # set directories to previously selected training patches
# dir_positives = f'{workdir}/foreground'
# dir_negatives = f'{workdir}/background'
#
# # extract feature data for all pixels in all patches
# # output is stores in .csv files in the same directories
# SegmentationFunctions.iterate_patches_dirs(dir_positives, dir_negatives)
#
# # ======================================================================================================================
# # (2) combine training data from all patches into single file
# # ======================================================================================================================
#
# # import all training data
# # get list of files
# files_pos = glob.glob(f'{dir_positives}/*.csv')
# files_neg = glob.glob(f'{dir_negatives}/*.csv')
# all_files = files_pos + files_neg
#
# # load data
# train_data = []
# for i, file in enumerate(all_files):
#     print(i)
#     data = pd.read_csv(file)
#     data = data.iloc[::10, :]  # only keep every 10th pixel of the patch
#     # data_mean = data.mean().to_frame().T
#     # data_mean['response'] = data['response'][0]
#     # train_data.append(data_mean)
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
# train_data = pd.read_csv(f'{workdir}/training_data.csv')
#
# # OPTIONAL: sample an equal number of rows per class for training
# n_pos = train_data.groupby('response').count().iloc[0, 0]
# n_neg = train_data.groupby('response').count().iloc[1, 0]
# n_min = min(n_pos, n_neg)
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
#
# # Use the random grid to search for best hyperparameters
# # First create the base model to tune
# rf = RandomForestClassifier()
# # Random search of parameters, using 3 fold cross validation,
# # search across 100 different combinations, and use all available cores
# rf_random = RandomizedSearchCV(estimator=rf,
#                                param_distributions= random_grid,
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
# param_grid = {
#     'bootstrap': [False],
#     'max_depth': [30, 40, 50, 60],
#     'max_features': [2, 4, 6, 8, 10],
#     'min_samples_leaf': [2, 4, 6, 8],
#     'min_samples_split': [8, 10, 12],
#     'n_estimators': [30, 40, 50, 60]
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
# specify model hyper-parameters
# clf = RandomForestClassifier(
#     max_depth=100,  # maximum depth of 95 decision nodes for each decision tree
#     max_features=8,  # maximum of 6 features (channels) are considered when forming a decision node
#     min_samples_leaf=6,  # minimum of 6 samples needed to form a final leaf
#     min_samples_split=10,  # minimum 4 samples needed to create a decision node
#     n_estimators=20,  # maximum of 55 decision trees
#     bootstrap=False,  # don't reuse samples
#     random_state=1,
#     n_jobs=-1
# )
#
# # fit random forest
# model = clf.fit(X, y)
# score = model.score(X, y)
#
# # save model
# path = f'{workdir}/Output/Models'
# if not Path(path).exists():
#     Path(path).mkdir(parents=True, exist_ok=True)
# pkl_filename = f'{path}/rf_segmentation.pkl'
# with open(pkl_filename, 'wb') as file:
#     pickle.dump(model, file)
#
#
# # ======================================================================================================================
# # (5) Get a tile of equal size from each patch
# # ======================================================================================================================
#
# import random
# import pandas as pd
#
# workdir = 'Z:/Public/Jonas/Data/ESWW006/Images_trainset'
#
# files = glob.glob(f"{workdir}/Patches/*.png")
#
# # sample the patches
# out_dir = Path(workdir) / "RectPatches"
# out_dir.mkdir(parents=True, exist_ok=True)
# for file in files:
#     img_name = os.path.basename(file)
#     stem_name = img_name.replace(".png", "")
#     out_name = f"{out_dir}/{stem_name}.JPG"
#     if not os.path.exists(out_name):
#         img = imageio.imread(file)
#         try:
#             y1 = random.randrange(0, img.shape[0] - 2400)
#             x1 = random.randrange(0, img.shape[1] - 2400)
#         except ValueError:
#             continue
#         y2 = y1 + 2400
#         x2 = x1 + 2400
#         patch = img[y1:y2, x1:x2, :]
#         coordinates = (x1, x2, y1, y2)
#         imageio.imwrite(out_name, patch)
#         coordinates = list(coordinates)
#         df = pd.DataFrame([coordinates])
#         df2 = df.set_axis(['x1', 'x2', 'y1', 'y2'], axis=1, inplace=False)
#         df2.to_csv(f'{out_dir}/{stem_name}.csv', index=False)
#     else:
#         print("output exists - skipping")
#         continue
#
# # ======================================================================================================================
# # (4) segment images
# # ======================================================================================================================
#
# workdir = 'Z:/Public/Jonas/Data/ESWW006/Images_trainset'
#
# from ImagePreSegmentor import ImagePreSegmentor
#
#
# def run():
#     dir_to_process = f'{workdir}/RectPatches'
#     dir_output = f'{workdir}/Output'
#     dir_model = f'{workdir}/Output/Models/rf_segmentation.pkl'
#     image_pre_segmentor = ImagePreSegmentor(dir_to_process=dir_to_process,
#                                             dir_model=dir_model,
#                                             dir_output=dir_output,
#                                             img_type="JPG",
#                                             overwrite=True)
#     image_pre_segmentor.segment_images()
#
#
# if __name__ == "__main__":
#     run()
#
# # ======================================================================================================================
# # 5. Create CVAT back-up
# # ======================================================================================================================
#
# from cvat_integration import BackupCreator
#
# workdir = 'Z:/Public/Jonas/Data/ESWW006/Images_trainset'
#
#
# def run():
#     dir_images = f"{workdir}/RectPatches"
#     dir_masks = f"{workdir}/Output/Mask"
#     dir_output = f"{workdir}/cvat_backup"
#     name = "cvat_backup"
#     img_type = ".JPG"
#     backup_creator = BackupCreator(dir_images=dir_images,
#                                    dir_masks=dir_masks,
#                                    dir_output=dir_output,
#                                    name=name,
#                                    img_type=img_type)
#     backup_creator.create()
#
#
# if __name__ == "__main__":
#     run()
#
#
# ======================================================================================================================
# 6. Extract soil patches
# ======================================================================================================================

import utils
from importlib import reload
reload(utils)
import pandas as pd

paths = glob.glob("Z:/Public/Jonas/Data/ESWW006/Images_trainset/*/JPEG/*.JPG")
parent_dir = Path("Z:/Public/Jonas/Data/ESWW006/Images_trainset")
meas_events = [x for x in os.listdir(parent_dir) if x.startswith("2022_")]

# loop over measurement events
for event in meas_events:

    # prepare workspace
    dir_event = parent_dir / event
    output_dir = dir_event / "Processed"
    checker_dir = output_dir / "CheckImages"
    patch_dir = output_dir / "SoilPatches"
    output_dir.mkdir(parents=True, exist_ok=True)
    checker_dir.mkdir(parents=True, exist_ok=True)
    patch_dir.mkdir(parents=True, exist_ok=True)

    # get all image paths
    plant_image_paths = glob.glob(f'{dir_event}/JPEG/ESWW*.JPG')
    soil_image_paths = glob.glob(f'{dir_event}/JPEG/soil*.JPG')  # rejects the plot images

    # loop over plant images
    for plant_path in plant_image_paths:

        plant_image_name = os.path.basename(plant_path).replace(".JPG", "")

        # loop over soil images
        for soil_path in soil_image_paths:

            image_name = os.path.basename(soil_path)
            image_png_name = image_name.replace(".JPG", ".png")

            image_original = cv2.imread(soil_path)
            image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB)

            # get reference
            target = plant_path
            pseudo_out_dir = "Z:/Public/Jonas/Data/ESWW006/Images_trainset/ColCorr"

            # ==============================================================================================================
            # COLOR CORRECTION
            # ==============================================================================================================

            corrected_patch_dir = patch_dir / plant_image_name / "Corrected"
            original_patch_dir = patch_dir / plant_image_name / "Original"
            corrected_patch_dir.mkdir(parents=True, exist_ok=True)
            original_patch_dir.mkdir(parents=True, exist_ok=True)

            corrected_patch_name = corrected_patch_dir / image_png_name
            original_patch_name = original_patch_dir / image_png_name

            if os.path.exists(original_patch_name):
                print("-- output exists - skipping...")
                continue
            try:
                image_corrected = utils.color_correction(filename_target=target,
                                                         filename_source=soil_path,
                                                         output_directory=pseudo_out_dir)
            except RuntimeError:
                soil_patch, coordinates = utils.get_soil_patch(image_original, size=(2400, 2400))
                x1, x2, y1, y2 = coordinates
                imageio.imwrite(original_patch_name, soil_patch)
                coordinates = list(coordinates)
                df = pd.DataFrame([coordinates])
                df2 = df.set_axis(['x1', 'x2', 'y1', 'y2'], axis=1, inplace=False)
                csv_namer = str(corrected_patch_name).replace(".png", ".csv")
                df2.to_csv(csv_namer, index=False)
                continue

            # ==============================================================================================================
            # SOIL PATCH EXTRACTION
            # ==============================================================================================================

            image_corrected_cut = image_corrected[:4500, :, :]
            soil_patch, coordinates = utils.get_soil_patch(image_original, size=(2400, 2400))
            x1, x2, y1, y2 = coordinates
            soil_patch_corrected = image_corrected_cut[y1:y2, x1:x2, :]

            # save output
            imageio.imwrite(corrected_patch_name, soil_patch_corrected)
            imageio.imwrite(original_patch_name, soil_patch)
            coordinates = list(coordinates)
            df = pd.DataFrame([coordinates])
            df2 = df.set_axis(['x1', 'x2', 'y1', 'y2'], axis=1, inplace=False)
            csv_namer = str(corrected_patch_name).replace(".png", ".csv")
            df2.to_csv(csv_namer, index=False)
            csv_namer = str(original_patch_name).replace(".png", ".csv")
            df2.to_csv(csv_namer, index=False)

# ======================================================================================================================

