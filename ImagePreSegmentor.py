
# ======================================================================================================================
# Author: Jonas Anderegg, jonas.anderegg@usys.ethz.ch
# Project: ColorTrends
# Use: Pre-segments images using a pre-trained model, saves masks, probability masks and overlays
# Last edited. 2022-04-12
# ======================================================================================================================


import SegmentationFunctions
import utils
from PIL import Image
import os
import glob
import pickle
import numpy as np
import cv2
import imageio
from pathlib import Path

from utils_smoothing import smooth_edge_aware


class ImagePreSegmentor:

    def __init__(self, dir_to_process, dir_output, dir_model, img_type, overwrite):
        self.dir_to_process = Path(dir_to_process)
        self.dir_model = Path(dir_model)
        # output paths
        self.path_output = Path(dir_output)
        self.path_mask = self.path_output / "Mask"
        self.path_proba = self.path_output / "Proba"
        self.path_overlay = self.path_output / "Overlay"
        self.path_solver = self.path_output / "Solver"
        self.image_type = img_type
        self.overwrite = overwrite

    def prepare_workspace(self):
        """
        Creates all required output directories
        """
        self.path_output.mkdir(parents=True, exist_ok=True)
        self.path_mask.mkdir(parents=True, exist_ok=True)
        self.path_proba.mkdir(parents=True, exist_ok=True)
        self.path_overlay.mkdir(parents=True, exist_ok=True)
        self.path_solver.mkdir(parents=True, exist_ok=True)

    def file_feed(self):
        """
        Creates a list of paths to images that are to be processed
        :param img_type: a character string, the file extension, e.g. "JPG"
        :return: paths
        """
        # get all files and their paths
        files = glob.glob(f'{self.dir_to_process}/*.{self.image_type}')
        return files

    def segment_image(self, img):
        """
        Segments an image using a pre-trained pixel classification model.
        Creates probability maps, binary segmentation masks, and overlay
        :param img: The image to be processed.
        :return: The resulting binary segmentation mask.
        """
        with open(self.dir_model, 'rb') as model:
            model = pickle.load(model)

        # extract pixel features
        color_spaces, descriptors, descriptor_names = SegmentationFunctions.get_color_spaces(img)
        descriptors_flatten = descriptors.reshape(-1, descriptors.shape[-1])

        # extract pixel label probabilities
        segmented_flatten_probs = model.predict_proba(descriptors_flatten)[:, 0]

        # restore image
        probabilities = segmented_flatten_probs.reshape((descriptors.shape[0], descriptors.shape[1]))

        # perform edge-aware smoothing
        output_solver, thresh = smooth_edge_aware(reference=img, target=probabilities)

        # smooth binary mask
        # TODO optimize the kernel size
        binary_mask = cv2.medianBlur(thresh.astype("uint8"), 7)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilate_th = cv2.dilate(binary_mask, kernel, iterations=1)

        # # Plot
        # fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
        # axs[0].imshow(dilate_th)
        # axs[0].set_title('img')
        # axs[1].imshow(out_mask)
        # axs[1].set_title('orig_mask')
        # plt.show(block=True)

        # TODO optimize the filters
        mask_filtered = utils.filter_objects_size(dilate_th, 150, "smaller")
        m_inv = cv2.bitwise_not(mask_filtered*255)
        mask_filtered = utils.filter_objects_size(m_inv, 150, "smaller")
        mask_filtered = cv2.bitwise_not(mask_filtered)

        # create overlay
        M = mask_filtered.ravel()
        M = np.expand_dims(M, -1)
        out_mask = np.dot(M, np.array([[1, 0, 0, 0.33]]))
        out_mask = np.reshape(out_mask, newshape=(img.shape[0], img.shape[1], 4))
        out_mask = out_mask.astype("uint8")
        mask = Image.fromarray(out_mask, mode="RGBA")
        img_ = Image.fromarray(img, mode="RGB")
        img_ = img_.convert("RGBA")
        img_.paste(mask, (0, 0), mask)
        overlay = np.asarray(img_)

        # plt.imshow(overlay)

        return probabilities, output_solver, mask_filtered, overlay

    def segment_images(self):
        """
        Wrapper, processing all images
        :param img_type: a character string, the file extension, e.g. "JPG"
        """
        self.prepare_workspace()
        files = self.file_feed()

        for file in files:

            # get file basename
            base_name = os.path.basename(file)

            print(base_name)

            png_name = base_name.replace("." + self.image_type, ".png")

            # output paths
            proba_name = self.path_proba / base_name
            mask_name = self.path_mask / png_name
            overlay_name = self.path_overlay / png_name
            solver_name = self.path_solver / png_name

            if not self.overwrite and os.path.exists(mask_name):
                continue
            else:
                img = imageio.imread(file)
                proba, solver, mask, overlay = self.segment_image(img)

                # save masks
                imageio.imwrite(mask_name, mask)
                imageio.imwrite(proba_name, proba)
                imageio.imwrite(overlay_name, overlay)
                imageio.imwrite(solver_name, solver)


