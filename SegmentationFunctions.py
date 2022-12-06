
# ======================================================================================================================
# Author: Jonas Anderegg, jonas.anderegg@usys.ethz.ch
# Functions for training data extraction: Pixel-wise segmentation of scanned leaf images into necrotic lesion and
# surrounding healthy tissue
# Last modified: 2021-11-10
# ======================================================================================================================

import glob
import imageio
import numpy as np
import cv2
import os
import pandas as pd
import random
import pickle


def get_color_spaces(patch):

    # Scale to 0...1
    img_RGB = np.array(patch / 255, dtype=np.float32)

    # Images are in RGBA mode, but alpha seems to be a constant - remove to convert to simple RGB
    img_RGB = img_RGB[:, :, :3]

    # Convert to other color spaces
    img_HSV = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HSV)
    img_Luv = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2Luv)
    img_Lab = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2Lab)
    img_YUV = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2YUV)
    img_YCbCr = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2YCrCb)

    # Calculate vegetation indices: ExR, ExG, TGI
    R, G, B = cv2.split(img_RGB)
    normalizer = np.array(R + G + B, dtype=np.float32)
    # Avoid division by zero
    normalizer[normalizer == 0] = 10
    r, g, b = (R, G, B) / normalizer

    # weights for TGI
    lambda_r = 670
    lambda_g = 550
    lambda_b = 480

    TGI = -0.5 * ((lambda_r - lambda_b) * (r - g) - (lambda_r - lambda_g) * (r - b))
    ExR = np.array(1.4 * r - b, dtype=np.float32)
    ExG = np.array(2.0 * g - r - b, dtype=np.float32)

    # Concatenate all
    descriptors = np.concatenate(
        [img_RGB, img_HSV, img_Lab, img_Luv, img_YUV, img_YCbCr, np.stack([ExG, ExR, TGI], axis=2)], axis=2)
    # Names
    descriptor_names = ['sR', 'sG', 'sB', 'H', 'S', 'V', 'L', 'a', 'b',
                        'L', 'u', 'v', 'Y', 'U', 'V', 'Y', 'Cb', 'Cr', 'ExG', 'ExR', 'TGI']

    # Return as tuple
    return (img_RGB, img_HSV, img_Lab, img_Luv, img_YUV, img_YCbCr, ExG, ExR, TGI), descriptors, descriptor_names


def extract_training_data(patch):

    color_spaces, descriptors, descriptor_names = get_color_spaces(patch)

    predictors = []

    # iterate over all pixels in the patch
    for x in range(patch.shape[0]):
        for y in range(patch.shape[1]):
            predictors_ = descriptors[x, y].tolist()
            # Append to training set
            predictors.append(predictors_)

    # Convert to numpy array
    a_predictors = np.array(predictors)

    return a_predictors, descriptor_names


def iterate_patches(image, mask):

    n_comps, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    X = []

    for comp in range(1, n_comps):

        print(comp)

        coords = np.where(output == comp)
        i = random.sample(range(0, len(coords[0])), 50)

        # randomly sample fifty px
        px = image[coords[1][i], coords[0][i], :]
        px_ = np.expand_dims(px, axis=0)
        color_spaces, descriptors, descriptor_names = get_color_spaces(px_)

        predictors = []

        # iterate over all pixels in the patch
        for x in range(px_.shape[0]):
            for y in range(px_.shape[1]):
                predictors_ = descriptors[x, y].tolist()
                # Append to training set
                predictors.append(predictors_)

        # Convert to numpy array
        a_predictors = np.array(predictors)
        X.append(a_predictors)

    X_all = np.vstack(X)
    df = pd.DataFrame(X_all, columns=descriptor_names)
    print(df)

    return df


def iterate_patches_dirs(dir_positives, dir_negatives):

    # POSITIVE patches
    all_files_pos = glob.glob(f'{dir_positives}/*.png')
    # iterate over patches
    for i, file in enumerate(all_files_pos):
        print(f'{i}/{len(all_files_pos)}')
        patch = imageio.imread(file)
        X, X_names = extract_training_data(patch)
        df = pd.DataFrame(X, columns=X_names)
        df['response'] = 'pos'
        f_name = os.path.splitext(os.path.basename(file))[0]
        df.to_csv(f'{dir_positives}/{f_name}_data.csv',  index=False)

    # NEGATIVE patches
    all_files_neg = glob.glob(f'{dir_negatives}/*.png')
    # iterate over patches
    for i, file in enumerate(all_files_neg):
        print(f'{i}/{len(all_files_neg)}')
        patch = imageio.imread(file)
        X, X_names = extract_training_data(patch)
        df = pd.DataFrame(X, columns=X_names)
        df['response'] = 'neg'
        f_name = os.path.splitext(os.path.basename(file))[0]
        df.to_csv(f'{dir_negatives}/{f_name}_data.csv',  index=False)


def iterate_patches_dirs2(dirs):
    for dir in dirs:
        all_files = glob.glob(f'{dir}/*.png')
        for i, file in enumerate(all_files):
            # print(f'{i}/{len(all_files)}')
            print(file)
            patch = imageio.imread(file)
            X, X_names = extract_training_data(patch)
            df = pd.DataFrame(X, columns=X_names)
            df['response'] = 'pos'
            f_name = os.path.splitext(os.path.basename(file))[0]
            df.to_csv(f'{dir}/{f_name}_data.csv', index=False)


def segment_image(image, dir_model):
    """
    Segments an image using a pre-trained pixel classification model.
    :param image: The image to be processed.
    :return: The resulting binary segmentation mask.
    """
    # print('-segmenting image')

    # load model
    with open(dir_model, 'rb') as model:
        model = pickle.load(model)

    # extract pixel features
    color_spaces, descriptors, descriptor_names = get_color_spaces(image)
    descriptors_flatten = descriptors.reshape(-1, descriptors.shape[-1])

    # predict pixel label
    segmented_flatten = model.predict(descriptors_flatten)

    # restore image, return as binary mask
    segmented = segmented_flatten.reshape((descriptors.shape[0], descriptors.shape[1]))
    segmented = np.where(segmented == 'pos', 1, 0)
    segmented = np.where(segmented == 0, 255, 0).astype("uint8")

    return segmented
