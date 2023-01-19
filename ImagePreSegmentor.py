
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
import pandas as pd
import cv2
import imageio
from pathlib import Path
import copy
import multiprocessing
from multiprocessing import Manager, Process

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

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


class ImagePostSegmentor:

    def __init__(self,
                 base_dir,
                 dirs_to_process,
                 dir_veg_masks,
                 dir_patch_coordinates,
                 dir_ear_masks, dir_output, dir_model, img_type, mask_type, overwrite, save_masked_images):
        self.path_base_dir = Path(base_dir)
        self.dirs_to_process = dirs_to_process
        self.dir_patch_coordinates = dir_patch_coordinates
        self.dir_veg_masks = dir_veg_masks
        self.dir_ear_masks = dir_ear_masks
        self.dir_model = Path(dir_model)
        # output paths
        self.path_output = Path(dir_output)
        self.path_mask = self.path_output / "Mask"
        # - color masks
        self.patch_mask_ear = self.path_output / "EarMask"
        self.patch_mask_veg = self.path_output / "VegMask"
        self.patch_mask_veg_no_ear = self.path_output / "VegMaskNoEar"
        # - image masks
        self.patch_img_veg = self.path_base_dir / "SegImg" / "VegMask"
        self.patch_img_ear = self.path_base_dir / "SegImg" / "EarMask"
        self.patch_img_veg_no_ear = self.path_base_dir / "SegImg" / "VegMaskNoEar"
        # - csv
        self.path_stats = self.path_output / "Stats"
        # helpers
        self.image_type = img_type
        self.mask_type = mask_type
        self.overwrite = overwrite
        self.save_masked_images = save_masked_images

    def prepare_workspace(self):
        """
        Creates all required output directories
        """
        for path in [self.path_output, self.path_mask, self.patch_mask_ear, self.patch_mask_veg,
                     self.patch_mask_veg_no_ear, self.patch_img_veg, self.patch_img_ear,
                     self.patch_img_veg_no_ear, self.path_stats]:
            path.mkdir(parents=True, exist_ok=True)

    def file_feed(self):
        """
        Creates a list of paths to images that are to be processed
        :param img_type: a character string, the file extension, e.g. "JPG"
        :return: paths
        """

        # get all files and their paths
        files = []
        for d in self.dirs_to_process:
            files.extend(glob.glob(f'{d}/*.{self.image_type}'))
        # removes all Reference images
        files = [f for f in files if "Ref" not in f]
        return files

    def segment_image(self, img):
        """
        Segments an image using a pre-trained pixel classification model.
        Creates probability maps, binary segmentation masks, and overlay
        :param veg_mask: vegetation mask (ground truth or flash predictions)
        :param img: The image to be processed.
        :return: The resulting binary segmentation mask.
        """
        with open(self.dir_model, 'rb') as model:
            model = pickle.load(model)

        model.n_jobs = 1  # disable parallel

        # extract pixel features
        color_spaces, descriptors, descriptor_names = SegmentationFunctions.get_color_spaces(img)
        descriptors_flatten = descriptors.reshape(-1, descriptors.shape[-1])

        # extract pixel label probabilities
        segmented_flatten_probs = model.predict(descriptors_flatten)

        # restore image
        preds = segmented_flatten_probs.reshape((descriptors.shape[0], descriptors.shape[1]))

        # convert to mask
        mask = np.zeros_like(img)
        mask[np.where(preds == "brown")] = (102, 61, 20)
        mask[np.where(preds == "yellow")] = (255, 204, 0)
        mask[np.where(preds == "green")] = (0, 100, 0)

        # fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
        # axs[0].imshow(mask)
        # axs[0].set_title('img')
        # axs[1].imshow(img)
        # axs[1].set_title('orig_mask')
        # plt.show(block=True)

        return mask

    def process_image(self, work_queue, result):

        for job in iter(work_queue.get, 'STOP'):

            file = job['file']

            # read image
            base_name = os.path.basename(file)
            stem_name = Path(file).stem
            png_name = base_name.replace("." + self.image_type, ".png")
            csv_name = base_name.replace("." + self.image_type, ".csv")

            img = Image.open(file)
            pix = np.array(img)

            # sample patch from image
            c = pd.read_table(f'{self.dir_patch_coordinates}/{stem_name}.txt', sep=",").iloc[0, :].tolist()
            patch = pix[c[2]:c[3], c[0]:c[1]]

            # downscale
            x_new = int(patch.shape[0] * (1 / 2))
            y_new = int(patch.shape[1] * (1 / 2))
            patch_seg = cv2.resize(patch, (y_new, x_new), interpolation=cv2.INTER_LINEAR)

            # output paths
            mask_name = self.path_mask / png_name

            if not self.overwrite and os.path.exists(mask_name):
                continue

            # segment the entire patch
            mask = self.segment_image(patch_seg)

            # upscale
            x_new = int(patch_seg.shape[0] * (2))
            y_new = int(patch_seg.shape[1] * (2))
            mask = cv2.resize(mask, (y_new, x_new), interpolation=cv2.INTER_NEAREST)

            # load segmentation masks
            veg_mask = imageio.imread(f'{self.dir_veg_masks}/{png_name}')
            ear_mask = imageio.imread(f'{self.dir_ear_masks}/{png_name}')

            # mask combinations
            veg_no_ear_mask = veg_mask - ear_mask
            veg_mask_two_mod = np.bitwise_or(veg_mask, ear_mask)

            # remove background and/or objects - color masks
            veg_col_mask = copy.copy(mask)
            veg_col_mask[np.where(veg_mask == 0)] = (0, 0, 0)

            ear_col_mask = copy.copy(mask)
            ear_col_mask[np.where(ear_mask == 0)] = (0, 0, 0)

            veg_col_mask_no_ear = copy.copy(veg_col_mask)
            veg_col_mask_no_ear[np.where(ear_mask == 255)] = (0, 0, 0)

            # remove background and/or objects - original patches
            veg_image = copy.copy(patch)
            veg_image[np.where(veg_mask == 0)] = (0, 0, 0)

            ear_image = copy.copy(patch)
            ear_image[np.where(ear_mask == 0)] = (0, 0, 0)

            veg_no_ear_image = copy.copy(veg_image)
            veg_no_ear_image[np.where(ear_mask == 255)] = (0, 0, 0)

            imageio.imwrite(mask_name, mask)
            imageio.imwrite(self.patch_mask_veg / png_name, veg_col_mask)
            imageio.imwrite(self.patch_mask_ear / png_name, ear_col_mask)
            imageio.imwrite(self.patch_mask_veg_no_ear / png_name, veg_col_mask_no_ear)
            if self.save_masked_images:
                imageio.imwrite(self.patch_img_veg / png_name, veg_image)
                imageio.imwrite(self.patch_img_ear / png_name, ear_image)
                imageio.imwrite(self.patch_img_veg_no_ear / png_name, veg_no_ear_image)

            # get color properties
            desc, desc_names = utils.color_index_transformation(veg_no_ear_image)
            df = pd.DataFrame()
            for d, d_n in zip(desc, desc_names):
                stats, stat_names = utils.index_distribution(image=patch, image_name=d_n, mask=mask)
                df[stat_names] = [stats]
            df.insert(loc=0, column='image_id', value=stem_name)

            # get pixel fractions
            # total cover per fraction
            veg_cover = len(np.where(veg_mask == 255)[0])/(4000*4000)
            ear_cover = len(np.where(ear_mask == 255)[0])/(4000*4000)
            veg_cover_2m = len(np.where(veg_mask_two_mod == 255)[0])/(4000*4000)
            veg_cover_no_ear = len(np.where(veg_no_ear_mask == 255)[0])/(4000*4000)
            cover_stat_names = ["veg_cover", "ear_cover", "veg_cover_2m", "veg_cover_no_ear"]
            df[cover_stat_names] = [[veg_cover, ear_cover, veg_cover_2m, veg_cover_no_ear]]

            # cover within fraction per color
            ear_green = len(np.where(ear_col_mask[:, :, 1] == 100)[0])/len(np.where(ear_mask == 255)[0])
            ear_chlr = len(np.where(ear_col_mask[:, :, 1] == 204)[0])/len(np.where(ear_mask == 255)[0])
            ear_necr = len(np.where(ear_col_mask[:, :, 1] == 61)[0])/len(np.where(ear_mask == 255)[0])
            veg_green = len(np.where(veg_col_mask[:, :, 1] == 100)[0])/len(np.where(veg_mask == 255)[0])
            veg_chlr = len(np.where(veg_col_mask[:, :, 1] == 204)[0])/len(np.where(veg_mask == 255)[0])
            veg_necr = len(np.where(veg_col_mask[:, :, 1] == 61)[0])/len(np.where(veg_mask == 255)[0])
            status_stat_names = ["ear_green", "ear_chlr", "ear_necr", "veg_green", "veg_chlr", "veg_necr"]
            df[status_stat_names] = [[ear_green, ear_chlr, ear_necr, veg_green, veg_chlr, veg_necr]]
            df.to_csv(self.path_stats / csv_name, index=False)
            result.put(file)

    def process_images(self):

        self.prepare_workspace()
        files = self.file_feed()

        if len(files) > 0:
            # make job and results queue
            m = Manager()
            jobs = m.Queue()
            results = m.Queue()
            processes = []
            # Progress bar counter
            max_jobs = len(files)
            count = 0

            # Build up job queue
            for file in files:
                print(file, "to queue")
                job = dict()
                job['file'] = file
                jobs.put(job)

            # Start processes
            for w in range(multiprocessing.cpu_count() - 10):
                p = Process(target=self.process_image,
                            args=(jobs, results))
                p.daemon = True
                p.start()
                processes.append(p)
                jobs.put('STOP')

            print(str(len(files)) + " jobs started, " + str(multiprocessing.cpu_count() - 10) + " workers")

            # Get results and increment counter along with it
            while count < max_jobs:
                img_names = results.get()
                count += 1
                print("processed " + str(count) + "/" + str(max_jobs))

            for p in processes:
                p.join()
