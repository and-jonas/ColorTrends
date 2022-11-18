import numpy as np
import cv2
import random
from plantcv import plantcv as pcv
import copy
import os


def random_patch(img, size, frame, random_patch=True):
    """
    Randomly select a patch from a defined central region of the image
    :param img: the RGB image to select the patch from
    :param size: the size of the patch in pixels
    :param frame: the size of the edge to be excluded; a tuple of four integers
    :return: the selected patch
    """
    # rotate image if necessary
    if img.shape[0] > img.shape[1]:
        print('rotating 90')
        img = np.rot90(img)

    # detect color checker, and rotate by 180 degrees if necessary
    try:
        dataframe1, start, space = pcv.transform.find_color_card(rgb_img=img, background='light')
    except RuntimeError:
        dataframe1, start, space = pcv.transform.find_color_card(rgb_img=img, background='dark')

    if start[1] < 2000:
        print('rotating 180')
        img = np.rot90(np.rot90(img))

    # remove edges
    left, upper, right, lower = frame
    img_central = img[upper:img.shape[0]-lower, left:img.shape[1]-right, :]

    # randomly select a patch
    if random_patch:
        y1 = random.randrange(0, img_central.shape[0]-size)
        x1 = random.randrange(0, img_central.shape[1]-size)
        y2 = y1 + size
        x2 = x1 + size
        img_patch = img_central[y1:y2, x1:x2, :]
        coords = (x1, x2, y1, y2)

    else:
        img_patch = img_central
        coords = None

    return img_patch, coords


def sample_random_patch(img, size, obstruction_mask=None):
    """
    Samples a random patch with defined size from an image.
    :param img: The image to sample the patch from
    :param size: a tuple defining the width and height of the patch
    :param obstruction_mask: if available, a binary mask with obstructions dominant. The patch will not be sampled
    from an image area containing obstructions.
    :return: the sampled patch and a tuple of coordinates defining where the patch was sampled
    """
    checker = False
    checker2 = 0

    while not checker and checker2 < 50:

        # randomly select a patch
        y1 = random.randrange(0, img.shape[0]-size[0])
        x1 = random.randrange(0, img.shape[1]-size[1])
        y2 = y1 + size[0]
        x2 = x1 + size[1]

        if obstruction_mask is not None:
            mask_patch = obstruction_mask[y1:y2, x1:x2]
            if (mask_patch != 0).any():
                checker = False
            else:
                checker = True
        else:
            checker = True

        checker2 += 1

    if not checker:
        return None, None
    else:
        img_patch = img[y1:y2, x1:x2, :]
        coordinates = (x1, x2, y1, y2)
        return img_patch, coordinates


def sample_patch_from_corner(image, patch_size=2400):
    """
    Crops a patch from the image, starting at a random image corner.
    :param image: the image to crop a patch from
    :param patch_size: the size of the patch
    :return: patch, image coordinates of two corners of the patch, a checker image with the rectangle marked
    """
    # randomly determine the corner
    corner = random.randrange(4)
    # fix the corner coordinates depending on the corner
    if corner == 0:
        y1 = 0
        x1 = 0
        y2 = y1 + patch_size
        x2 = x1 + patch_size
    elif corner == 1:
        y2 = image.shape[0]
        y1 = y2 - patch_size
        x1 = 0
        x2 = x1 + patch_size
    elif corner == 2:
        y2 = image.shape[0]
        y1 = y2 - patch_size
        x2 = image.shape[1]
        x1 = x2 - patch_size
    elif corner == 3:
        y1 = 0
        y2 = y1 + patch_size
        x2 = image.shape[1]
        x1 = x2 - patch_size
        x2 = x1 + patch_size

    # crop the image and extract the coordinates
    patch = image[y1:y2, x1:x2, :]
    coordinates = (x1, x2, y1, y2)

    # generate a checker image
    checker = copy.copy(image)
    cv2.rectangle(checker, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0), thickness=5)

    return patch, coordinates, checker, corner


def sample_patch_chessboard(image, patch_size=2400):
    """
    Crops a patch from the image, starting at a random image corner.
    :param image: the image to crop a patch from
    :param patch_size: the size of the patch
    :return: patch, image coordinates of two corners of the patch, a checker image with the rectangle marked
    """

    size_x, size_y = image.shape[:2]
    div = int(np.floor(size_x/patch_size))
    n_patches = div * div
    stride = patch_size + int((size_x - div*patch_size)/(div-1))

    idx = random.randrange(n_patches)
    row = int(np.floor(idx / div))
    pos = idx % div

    p = image[row * stride:row * stride + patch_size, pos * stride:pos * stride + patch_size]
    c = (row * stride, pos * stride, row * stride + patch_size, pos * stride + patch_size)

    checker = copy.copy(image)
    cv2.rectangle(checker, pt1=(c[1], c[0]), pt2=(c[3], c[2]), color=(255, 0, 0), thickness=9)

    return p, c, checker, idx


def color_correction(filename_target, filename_source, output_directory):

    print("- performing color correction...")

    # read target image and transform to RGB
    target_img, targetpath, targetname = pcv.readimage(filename=filename_target)
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

    # rotate image if not landscape
    if target_img.shape[0] > target_img.shape[1]:
        target_img = np.rot90(np.rot90(np.rot90(target_img)))

    # detect the color checker
    try:
        dataframe1, start, space = pcv.transform.find_color_card(rgb_img=target_img, background='light')
    except:
        dataframe1, start, space = pcv.transform.find_color_card(rgb_img=target_img, background='dark')
    target_mask = pcv.transform.create_color_card_mask(target_img, radius=15, start_coord=start,
                                                       spacing=space, nrows=4, ncols=6)

    # read source image and transform to RGB
    source_img, sourcepath, sourcename = pcv.readimage(filename=filename_source)
    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)

    # rotate image if not landscape
    if source_img.shape[0] > source_img.shape[1]:
        source_img = np.rot90(np.rot90(np.rot90(source_img)))

    # detect the color checker
    try:
        dataframe1, start, space = pcv.transform.find_color_card(rgb_img=source_img, background='light')
    except:
        dataframe1, start, space = pcv.transform.find_color_card(rgb_img=source_img, background='dark')

    source_mask = pcv.transform.create_color_card_mask(source_img, radius=10, start_coord=start,
                                                       spacing=space, nrows=4, ncols=6)

    # perform the correction
    tm, sm, transformation_matrix, corrected_img = pcv.transform.correct_color(target_img=target_img,
                                                                               target_mask=target_mask,
                                                                               source_img=source_img,
                                                                               source_mask=source_mask,
                                                                               output_directory=output_directory)

    return corrected_img


def get_soil_patch(image, size):

    print(" - extracting patch...")

    image_cut = image[:4500, :, :]

    # detect vegetation
    exg = index_TGI(image_cut)
    mask = exg > 4.5
    mask = np.uint8(np.where(mask, 1, 0))
    mask_filtered = filter_objects_size(mask, 2000, "smaller")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closing = cv2.morphologyEx(mask_filtered, cv2.MORPH_CLOSE, kernel)
    mask_veg = filter_objects_size(closing, 25000, "smaller")

    # detect shadowed areas
    grey = cv2.cvtColor(image_cut, cv2.COLOR_RGB2GRAY)
    mask = grey < 50
    mask = np.uint8(np.where(mask, 1, 0))
    mask = cv2.medianBlur(mask, 31)
    mask_shadows = filter_objects_size(mask, 25000, "smaller")

    # detect metal bars
    thresh = np.percentile(grey, 98.5)
    mask = grey > thresh
    mask = np.uint8(np.where(mask, 1, 0))
    mask = cv2.medianBlur(mask, 31)
    mask_bars = filter_objects_size(mask, 35000, "smaller")

    # combine masks
    combined_mask = mask_veg | mask_shadows | mask_bars
    cnts, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(combined_mask)
    for c in cnts:
        cv2.drawContours(final_mask, c, -1, color=1, thickness=150)

    # sample a 2500 x 2500 patch from usable image area
    soil_patch, coordinates = sample_random_patch(image_cut,
                                                  size=size,
                                                  obstruction_mask=combined_mask)
    return soil_patch, coordinates


def binarize_mask(mask):
    mask_green = mask[:, :, 1]
    plant_id = np.max(np.unique(mask_green))
    mask_binary = np.where(mask_green == plant_id, 255, 0).astype("uint8")
    return mask_binary


def image_tiler(image, stride):

    if image.shape[0] % stride == 0 and image.shape[1] % stride == 0:
        currentx = 0
        currenty = 0
        tiles = []
        while currenty < image.shape[1]:
            while currentx < image.shape[0]:
                print(currentx, ",", currenty)
                tile = image[currentx:currentx + stride, currenty:currenty + stride]
                # tile = image.crop((currentx, currenty, currentx + stride, currenty + stride))
                tiles.append(tile)
                currentx += stride
            currenty += stride
            currentx = 0

        return tuple(tiles)

    else:
        print("sorry your image does not fit neatly into", stride, "*", stride, "tiles")

        return None


def sample_patches(image, mask, size):

    img_size = image.shape[0]

    # image
    patch1 = image[:size, :size]
    patch2 = image[:size, img_size-size:img_size]
    patch3 = image[img_size-size:img_size, :size]
    patch4 = image[img_size-size:img_size, img_size-size:img_size]

    # mask
    mask1 = mask[:size, :size]
    mask2 = mask[:size, img_size-size:img_size]
    mask3 = mask[img_size-size:img_size, :size]
    mask4 = mask[img_size-size:img_size, img_size-size:img_size]

    return (patch1, patch2, patch3, patch4), (mask1, mask2, mask3, mask4)


def filter_objects_size(mask, size_th, dir):
    """
    Filter objects in a binary mask by size
    :param mask: A binary mask to filter
    :param size_th: The size threshold used to filter (objects GREATER than the threshold will be kept)
    :return: A binary mask containing only objects greater than the specified threshold
    """
    _, output, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    sizes = stats[1:, -1]
    if dir == "greater":
        idx = (np.where(sizes > size_th)[0] + 1).tolist()
    if dir == "smaller":
        idx = (np.where(sizes < size_th)[0] + 1).tolist()
    out = np.in1d(output, idx).reshape(output.shape)
    cleaned = np.where(out, 0, mask)

    return cleaned


# Function to extract RGB intensity histograms for images
def get_hist(img, normalize):

    # extract histograms
    cols = ('r', 'g', 'b')
    hist = []
    # loop over color channels
    for i, col in enumerate(cols):
        # get intensity histogram
        histo = cv2.calcHist(img, [i], None, [256], [0, 256])
        # normalize histogram (so that image size does not matter for clf)
        if normalize:
            histo = np.true_divide(histo, histo.sum())
        hist.append(histo)

    # get concatenated histograms in long format
    h_df = np.concatenate(hist, axis=0)
    out = np.transpose(h_df)

    return out

# =====

# vegetation index
def calculate_Index(img):
    # Calculate vegetation indices: ExR, ExG, TGI
    R, G, B = cv2.split(img)

    normalizer = np.array(R.astype("float32") + G.astype("float32") + B.astype("float32"))

    # Avoid division by zero
    normalizer[normalizer == 0] = 1
    r, g, b = (R, G, B) / normalizer

    ExG = np.array(2.0 * g - r - b, dtype=np.float32)

    return ExG

# =====


def flatten_contour_data(input_contour, asarray, as_point_list=True):
    """
    Extract contour points from cv2 format into point list
    :param input_contour: The cv2 contour to extract
    :param asarray: Boolean, whether output should be returned as an array
    :param as_point_list: Boolean, whetheer output should be returned as a point list
    :return: array or list containing the contour point coordinate pairs
    """
    xs = []
    ys = []
    for point in input_contour[0]:
        x = point[0][1]
        y = point[0][0]
        xs.append(x)
        ys.append(y)
    if as_point_list:
        point_list = []
        # for a, b in zip(xs, ys):
        for a, b in zip(ys, xs):
            point_list.append([a, b])
            c = point_list
        if asarray:
            c = np.asarray(point_list)
        return c
    else:
        return xs, ys


def make_point_list(input):
    """
    Transform cv2 format to ordinary point list
    :param input:
    :return: list of point coordinates
    """
    xs = []
    ys = []
    for point in range(len(input)):
        x = input[point]
        y = input[point]
        xs.append(x)
        ys.append(y)
    point_list = []
    for a, b in zip(xs, ys):
        point_list.append([a, b])
    c = point_list

    return c


def index_TGI(image):

    image = np.float32(image)

    R, G, B = cv2.split(image)

    normalizer = np.array(R + G + B, dtype=np.float32)
    # Avoid division by zero
    normalizer[normalizer == 0] = 1
    r, g, b = (R, G, B) / normalizer

    lambda_R = 670
    lambda_G = 550
    lambda_B = 480

    rTGI = -0.5 * ((lambda_R - lambda_B) * (r - g) - (lambda_R - lambda_G) * (r - b))

    return rTGI


def get_plot(name):

    n = os.path.basename(name).replace(".png", "")
    id = n.split("_")[0:2]
    id = "_".join(id)

    return id


def get_soil_id(name):

    n = os.path.basename(name).replace(".png", "")
    id = n.split("_")[2:]
    id = "_".join(id)

    return id


def apply_intensity_map(image, intensity_map):

    r, g, b = cv2.split(image)
    r_ = r * intensity_map
    g_ = g * intensity_map
    b_ = b * intensity_map
    r_ = np.where(r_ > 255, 255, r_)
    g_ = np.where(g_ > 255, 255, g_)
    b_ = np.where(b_ > 255, 255, b_)
    img_ = cv2.merge((r_, g_, b_))
    img_ = np.uint8(img_)

    return img_


# OPTIONAL: select n soil scenario
def get_identifier(file_names):
    ids = []
    for name in file_names:
        n = os.path.basename(name).replace(".png", "")
        id = n.split("_")[0:2]
        id = "_".join(id)
        ids.append(id)
    return ids


def get_plot_id(file_names):
    plot_ids = []
    for name in file_names:
        n = os.path.basename(name)
        plot_ids.append(n.split("_")[0])
    return plot_ids