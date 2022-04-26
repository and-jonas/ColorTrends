import os.path

import imageio
import cv2
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import utils
import json
import jsonlines
import glob


mask_paths = glob.glob("Z:/Public/Jonas/003_ESWW/ColorTrends/testing_segmentation_outdoor/Output/Mask/*.png")

polygons = []
metas = [{"version": "1.1"}, {"type": "images"}]
task = {"name":"test_backup_outdoor","bug_tracker":"","status":"annotation","labels":[{"name":"plant","color":"#3df53d","attributes":[]},{"name":"background","color":"#fa3253","attributes":[]}],"subset":"","version":"1.0","data":{"chunk_size":45,"image_quality":70,"start_frame":0,"stop_frame":(len(mask_paths)-1),"storage_method":"cache","storage":"local","sorting_method":"lexicographical","chunk_type":"imageset"},"jobs":[{"start_frame":0,"stop_frame":(len(mask_paths)-1),"status":"annotation"}]}
index = {}
keys = range(len(mask_paths))
values = [k*95+36 for k in range(len(mask_paths))]
for i in keys:
    index[i] = values[i]

for i, mask_path in enumerate(mask_paths):

    mask = imageio.imread(mask_path)

    # meta
    base_name = os.path.basename(mask_path)
    stem_name = base_name.split(("."))[0]
    meta = {"name": stem_name, "extension": ".JPG",
            "width": mask.shape[1], "height": mask.shape[0],
            "meta":{"related_images": []}}
    metas.append(meta)

    # add border
    mask[0, :] = 255
    mask[mask.shape[0] - 1, :] = 255
    mask[:, 0] = 255
    mask[:, mask.shape[1] - 1] = 255
    # plt.imshow(mask)

    mask_ = np.bitwise_not(mask)
    # plt.imshow(mask_)

    contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # # ignore the first contour, which enframes the entire image
    # contours = contours[1:]
    # hier = hier[0][1:]

    # extract contour points for each contour and write to dictionary
    cimg = np.zeros_like(mask)
    for h, c in zip(hier[0], contours):
        contour_points = utils.flatten_contour_data([c], asarray=True)
        if len(contour_points) < 3:
            continue
        else:
            # contour_points = contour_points[::5]
            cp = contour_points.flatten().tolist()
            if h[3] == -1:
                label = "background"
            else:
                label = "plant"
            polygon = {"type": "polygon", "occluded": "false", "z_order": 0,
                       "rotation": 0.0, "points": cp, "frame": i, "group": 0, "source": "manual", "attributes": [],
                       "label": label}
            polygons.append(polygon)
            if h[3] == -1:
                cv2.drawContours(cimg, c, -1, color=255, thickness=-1)
            else:
                cv2.drawContours(cimg, c, -1, color=112, thickness=-1)

    # plt.imshow(cimg)

data_set = [{"version": 0, "tags": [], "shapes": polygons, "tracks": []}]

with open("Z:/Public/Jonas/003_ESWW/test_backup_outdoor/annotations" + '.json', 'w') as json_file:
    json.dump(data_set, json_file)

with open("Z:/Public/Jonas/003_ESWW/test_backup_outdoor/task" + '.json', 'w') as json_file:
    json.dump(task, json_file)

with open("Z:/Public/Jonas/003_ESWW/test_backup_outdoor/data/index" + '.json', 'w') as json_file:
    json.dump(index, json_file)

# TODO Fix export of the manifest
# with open("Z:/Public/Jonas/003_ESWW/test_backup_outdoor/data/manifest" + ".jsonl", 'w') as f:
#     for item in metas:
#         f.write(json.dumps(item) + "\n")
#
# with jsonlines.open("Z:/Public/Jonas/003_ESWW/test_backup_outdoor/data/manifest" + ".jsonl", 'w') as writer:
#     writer.write_all(metas)
#
# with open("Z:/Public/Jonas/003_ESWW/test_backup_outdoor/data/manifest" + ".jsonl", 'w') as outfile:
#     for entry in metas:
#         json.dump(entry, outfile)
#         outfile.write('\n')
