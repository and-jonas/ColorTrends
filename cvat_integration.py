import os.path
import imageio
import cv2
import numpy as np
import utils
import json
import jsonlines
import glob
from pathlib import Path
import shutil


class BackupCreator:

    def __init__(self, dir_images, dir_masks, name, dir_output, img_type):
        self.path_images = Path(dir_images)
        self.path_masks = Path(dir_masks)
        self.name = name
        self.path_output = Path(dir_output)
        self.path_data = self.path_output / "data"
        self.img_type = img_type

    def prepare_workspace(self):
        """
        Creates all required output directories
        """
        self.path_output.mkdir(parents=True, exist_ok=True)
        self.path_data.mkdir(parents=True, exist_ok=True)

    def create_cvat_backup(self):
        """
        Creates a cvat backup task with all necessary files, except the manifest.
        The manifest still needs to be added manually due to a problem with writing of the jsonlines file.
        """
        # get all available masks
        mask_paths = glob.glob(f"{self.path_masks}/*.png")

        # initialize outputs
        polygons = []
        metas = [{"version": "1.1"}, {"type": "images"}]
        task = {"name":self.name,"bug_tracker":"","status":"annotation","labels":[{"name":"plant","color":"#3df53d","attributes":[]},{"name":"background","color":"#fa3253","attributes":[]}],"subset":"","version":"1.0","data":{"chunk_size":45,"image_quality":70,"start_frame":0,"stop_frame":(len(mask_paths)-1),"storage_method":"cache","storage":"local","sorting_method":"lexicographical","chunk_type":"imageset"},"jobs":[{"start_frame":0,"stop_frame":(len(mask_paths)-1),"status":"annotation"}]}
        index = {}
        keys = range(len(mask_paths))
        values = [k*95+36 for k in range(len(mask_paths))]
        for i in keys:
            index[i] = values[i]

        # iterate over all masks
        print("processing masks...")
        for i, mask_path in enumerate(mask_paths):

            print(i)

            # load mask
            mask = imageio.imread(mask_path)

            # meta
            base_name = os.path.basename(mask_path)
            stem_name = base_name.split(".")[0]
            meta = {"name": stem_name, "extension": self.img_type,
                    "width": mask.shape[1], "height": mask.shape[0],
                    "meta": {"related_images": []}}
            metas.append(meta)

            # add a border to obtain soil segments also for border regions
            mask[0, :] = 255
            mask[mask.shape[0] - 1, :] = 255
            mask[:, 0] = 255
            mask[:, mask.shape[1] - 1] = 255

            # get segment contours
            contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            # ignore the first contour, which frames the entire image
            contours = contours[1:]
            hier = hier[0][1:]

            # extract contour points for each contour and write to dictionary
            cimg = np.zeros_like(mask)
            for h, c in zip(hier, contours):
                contour_points = utils.flatten_contour_data([c], asarray=True)
                if len(contour_points) < 3:
                    continue
                else:
                    cp = contour_points.flatten().tolist()
                    if h[3] == -1:
                        label = "background"
                    else:
                        label = "plant"
                    polygon = {"type": "polygon", "occluded": "false", "z_order": 0,
                               "rotation": 0.0, "points": cp, "frame": i, "group": 0, "source": "manual", "attributes": [],
                               "label": label}
                    polygons.append(polygon)
                    # automatically classify the contours based on their hierarchy
                    if h[3] == -1:
                        cv2.drawContours(cimg, c, -1, color=255, thickness=-1)
                    else:
                        cv2.drawContours(cimg, c, -1, color=112, thickness=-1)

        # assemble all annotations
        annotations = [{"version": 0, "tags": [], "shapes": polygons, "tracks": []}]

        # write output
        print("writing output files...")
        with open(self.path_output / "annotations.json", 'w') as json_file:
            json.dump(annotations, json_file)
        with open(self.path_output / "task.json", 'w') as json_file:
            json.dump(task, json_file)
        with open(self.path_data / "index.json", 'w') as json_file:
            json.dump(index, json_file)
        with open(self.path_data / "manifest.jsonl", 'w', encoding='utf-8') as f:
            for line in metas:
                json_record = json.dumps(line, ensure_ascii=False)
                f.write(json_record + '\n')

        # copy the images to the backup data folder
        paths_images = glob.glob(f"{self.path_images}/*{self.img_type}")
        print("copying images...")
        for p in paths_images:
            image_base_name = os.path.basename(p)
            dir_to = self.path_data / image_base_name
            shutil.copy(p, dir_to)

        print("done")

    def create(self):
        self.prepare_workspace()
        self.create_cvat_backup()

