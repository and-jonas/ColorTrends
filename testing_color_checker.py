
import cv2
import matplotlib as mpl
mpl.use('Qt5Agg')
from plantcv import plantcv as pcv
import glob
import os
import imageio
import numpy as np
import utils
from importlib import reload
reload(utils)
import pandas as pd
import multiprocessing
from multiprocessing import Manager, Process
from pathlib import Path


class ColorCorrector:

    def __init__(self, base_dir, output_dir):

        self.base_dir = base_dir
        self.output_dir = output_dir
        self.list_of_image_paths = glob.glob("O:/Evaluation/FIP/2013/WW002/RGB/*-108/JPG/*.JPG")
        self.list_of_image_paths = [k for k in self.list_of_image_paths if '2013-06-13' not in k]

    def process_image(self, work_queue, result):

        # get reference
        target = f"{self.base_dir}target_midlc.JPG"

        # read images and transform to RGB
        target_img, targetpath, targetname = pcv.readimage(filename=target)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

        dataframe1, start, space = pcv.transform.find_color_card(rgb_img=target_img, background='light')

        target_mask = pcv.transform.create_color_card_mask(target_img, radius=15, start_coord=start,
                                                           spacing=space, nrows=4, ncols=6)

        # # Plot result
        # fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
        # # Show RGB and segmentation mask
        # axs[0].imshow(target_mask)
        # axs[0].set_title('img')
        # axs[1].imshow(target_img)
        # axs[1].set_title('orig_mask')
        # plt.show(block=True)

        exg = []
        exg_n = []

        for job in iter(work_queue.get, 'STOP'):

            image_name = job['image_name']
            img_name = image_name
            image_path = job['image_path']

            try:

                base_name = os.path.basename(image_path)
                base_name = base_name.replace(".JPG", ".png")
                source_img, sourcepath, sourcename = pcv.readimage(filename=image_path)
                source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)

                try:
                    dataframe1, start, space = pcv.transform.find_color_card(rgb_img=source_img, background='light')
                except:
                    dataframe1, start, space = pcv.transform.find_color_card(rgb_img=source_img, background='dark')

                source_mask = pcv.transform.create_color_card_mask(source_img, radius=10, start_coord=start,
                                                                   spacing=space, nrows=4, ncols=6)

                tm, sm, transformation_matrix, corrected_img = pcv.transform.correct_color(target_img=target_img,
                                                                                           target_mask=target_mask,
                                                                                           source_img=source_img,
                                                                                           source_mask=source_mask,
                                                                                           output_directory=self.output_dir)

                imageio.imwrite(f"{self.output_dir}/{base_name}", corrected_img)

                # ==================================================================================================================
                # extract greenness indicator from greenest 33% of pixels in central region of image
                # ==================================================================================================================

                rois = [source_img[500:2200, 1700:4200], corrected_img[500:2200, 1700:4200]]

                for i, roi in enumerate(rois):
                    # mask for paper background
                    upper_black = np.array([25, 25, 25], dtype=np.uint8)  # threshold for white pixels
                    lower_black = np.array([0, 0, 0], dtype=np.uint8)
                    mask_inv = cv2.inRange(roi, lower_black, upper_black)  # could also use threshold
                    mask = np.bitwise_not(mask_inv)
                    mask_pp = cv2.medianBlur(mask, 7)

                    t_exg = utils.calculate_Index(roi) * mask_pp

                    idx = np.where(t_exg != 0)
                    values = t_exg[idx]
                    # sorted array
                    sorted_index_array = np.argsort(values)
                    # sorted array
                    sorted_array = values[sorted_index_array]
                    # take n largest value
                    rslt = sorted_array[-1402500:]
                    # take mean
                    mean_exg_top33 = np.mean(rslt)
                    if i == 0:
                        exgreen = mean_exg_top33
                        exg.append(mean_exg_top33)
                    elif i == 1:
                        exgreen_n = mean_exg_top33
                        exg_n.append(mean_exg_top33)

                # with open("Z:/Public/Jonas/003_ESWW/ColorTrends/train_img_selection/exg_iter.csv", 'wt') as f:
                #     writer = csv.writer(f)
                #     writer.writerow(('img_id', 'exg', 'exg_n'))
                #     row = (base_name, exgreen, exgreen_n)
                #     writer.writerow(row)

                df = pd.DataFrame({'img_id': base_name, 'exg': exgreen, 'exg_n': exgreen_n}, index=[0])
                df = pd.DataFrame(df)
                df.to_csv(f'Z:/Public/Jonas/003_ESWW/ColorTrends/train_img_selection/exgr_iter/{image_name}_exgr.csv')

                # base_names.append(base_name)

                result.put(img_name)

            except Exception as e:
                print(image_path)
                print(e)
                continue

    def run(self):

        files = self.list_of_image_paths

        image_paths = {}
        for i, file in enumerate(files):
            image_name = Path(file).stem

            # test if already processed - skip
            # final_output_path = self.path_num_output_name / (image_name + '.csv')
            # if final_output_path.exists():
            #     continue
            # otherwise add to jobs list
            # else:
            image_path = file
            image_paths[image_name] = image_path

        if len(image_paths) > 0:
            # make job and results queue
            m = Manager()
            jobs = m.Queue()
            results = m.Queue()
            processes = []
            # Progress bar counter
            max_jobs = len(image_paths)
            count = 0

            # Build up job queue
            for image_name, image_path in image_paths.items():
                print(image_name, "to queue")
                job = dict()
                job['image_name'] = image_name
                job['image_path'] = image_path
                jobs.put(job)

            # Start processes
            for w in range(multiprocessing.cpu_count() - 1):
                p = Process(target=self.process_image,
                            args=(jobs, results))
                p.daemon = True
                p.start()
                processes.append(p)
                jobs.put('STOP')

            print(str(len(image_paths)) + " jobs started, " + str(multiprocessing.cpu_count() - 9) + " workers")

            # Get results and increment counter along with it
            while count < max_jobs:
                img_names = results.get()
                count += 1
                print("processed " + str(count) + "/" + str(max_jobs))

            for p in processes:
                p.join()

# for iter, img in enumerate(source_images):
#
#     # ==================================================================================================================
#     # perform color correction for all images
#     # ==================================================================================================================
#
#     base_name = os.path.basename(img)
#     base_name = base_name.replace(".JPG", ".png")
#     source_img, sourcepath, sourcename = pcv.readimage(filename=img)
#     source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
#
#     try:
#         dataframe1, start, space = pcv.transform.find_color_card(rgb_img=source_img, background='light')
#     except:
#         dataframe1, start, space = pcv.transform.find_color_card(rgb_img=source_img, background='dark')
#
#     source_mask = pcv.transform.create_color_card_mask(source_img, radius=10, start_coord=start,
#                                                        spacing=space, nrows=4, ncols=6)
#
#     tm, sm, transformation_matrix, corrected_img = pcv.transform.correct_color(target_img=target_img,
#                                                                                target_mask=target_mask,
#                                                                                source_img=source_img,
#                                                                                source_mask=source_mask,
#                                                                                output_directory=output_dir)
#
#     # ==================================================================================================================
#     # extract greenness indicator from greenest 33% of pixels in central region of image
#     # ==================================================================================================================
#
#     rois = [source_img[500:2200, 1700:4200], corrected_img[500:2200, 1700:4200]]
#
#     for i, roi in enumerate(rois):
#         # mask for paper background
#         upper_black = np.array([25, 25, 25], dtype=np.uint8)  # threshold for white pixels
#         lower_black = np.array([0, 0, 0], dtype=np.uint8)
#         mask_inv = cv2.inRange(roi, lower_black, upper_black)  # could also use threshold
#         mask = np.bitwise_not(mask_inv)
#         mask_pp = cv2.medianBlur(mask, 7)
#
#         t_exg = utils.calculate_Index(roi) * mask_pp
#
#         idx = np.where(t_exg != 0)
#         values = t_exg[idx]
#         # sorted array
#         sorted_index_array = np.argsort(values)
#         # sorted array
#         sorted_array = values[sorted_index_array]
#         # take n largest value
#         rslt = sorted_array[-1402500:]
#         # take mean
#         mean_exg_top33 = np.mean(rslt)
#         if i == 0:
#             exgreen = mean_exg_top33
#             exg.append(mean_exg_top33)
#         elif i == 1:
#             exgreen_n = mean_exg_top33
#             exg_n.append(mean_exg_top33)
#
#     with open("Z:/Public/Jonas/003_ESWW/ColorTrends/train_img_selection/exg_iter.csv", 'wt') as f:
#         writer = csv.writer(f)
#         writer.writerow(('img_id', 'exg', 'exg_n'))
#         row = (base_name, exgreen, exgreen_n)
#         writer.writerow(row)
#
#     base_names.append(base_name)
#
# df = ({'img_id': base_names, 'exg': exg, 'exg_n': exg_n})
# df = pd.DataFrame(df)
# df.to_csv("Z:/Public/Jonas/003_ESWW/ColorTrends/train_img_selection/exg.csv")

# ======================================================================================================================
