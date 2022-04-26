import numpy as np
import cv2
import random
from plantcv import plantcv as pcv


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


def random_soil_patch(img, size):

    # randomly select a patch
    y1 = random.randrange(0, img.shape[0]-size[0])
    x1 = random.randrange(0, img.shape[1]-size[1])
    y2 = y1 + size[0]
    x2 = x1 + size[1]

    img_patch = img[y1:y2, x1:x2, :]
    coords = (x1, x2, y1, y2)

    return img_patch, coords


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
    _, output, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
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

# =====

# image quilting

import numpy as np
import math
from skimage import io, util
import heapq


def randomPatch(texture, patchLength):
    h, w, _ = texture.shape
    i = np.random.randint(h - patchLength)
    j = np.random.randint(w - patchLength)

    return texture[i:i + patchLength, j:j + patchLength]


def L2OverlapDiff(patch, patchLength, overlap, res, y, x):
    error = 0

    if x > 0:
        left = patch[:, :overlap] - res[y:y + patchLength, x:x + overlap]
        error += np.sum(left ** 2)

    if y > 0:
        up = patch[:overlap, :] - res[y:y + overlap, x:x + patchLength]
        error += np.sum(up ** 2)

    if x > 0 and y > 0:
        corner = patch[:overlap, :overlap] - res[y:y + overlap, x:x + overlap]
        error -= np.sum(corner ** 2)

    return error


def randomBestPatch(texture, patchLength, overlap, res, y, x):
    h, w, _ = texture.shape
    errors = np.zeros((h - patchLength, w - patchLength))

    for i in range(h - patchLength):
        for j in range(w - patchLength):
            patch = texture[i:i + patchLength, j:j + patchLength]
            e = L2OverlapDiff(patch, patchLength, overlap, res, y, x)
            errors[i, j] = e

    i, j = np.unravel_index(np.argmin(errors), errors.shape)
    return texture[i:i + patchLength, j:j + patchLength]


def minCutPath(errors):
    # dijkstra's algorithm vertical
    pq = [(error, [i]) for i, error in enumerate(errors[0])]
    heapq.heapify(pq)

    h, w = errors.shape
    seen = set()

    while pq:
        error, path = heapq.heappop(pq)
        curDepth = len(path)
        curIndex = path[-1]

        if curDepth == h:
            return path

        for delta in -1, 0, 1:
            nextIndex = curIndex + delta

            if 0 <= nextIndex < w:
                if (curDepth, nextIndex) not in seen:
                    cumError = error + errors[curDepth, nextIndex]
                    heapq.heappush(pq, (cumError, path + [nextIndex]))
                    seen.add((curDepth, nextIndex))


def minCutPath2(errors):
    # dynamic programming, unused
    errors = np.pad(errors, [(0, 0), (1, 1)],
                    mode='constant',
                    constant_values=np.inf)

    cumError = errors[0].copy()
    paths = np.zeros_like(errors, dtype=int)

    for i in range(1, len(errors)):
        M = cumError
        L = np.roll(M, 1)
        R = np.roll(M, -1)

        # optimize with np.choose?
        cumError = np.min((L, M, R), axis=0) + errors[i]
        paths[i] = np.argmin((L, M, R), axis=0)

    paths -= 1

    minCutPath = [np.argmin(cumError)]
    for i in reversed(range(1, len(errors))):
        minCutPath.append(minCutPath[-1] + paths[i][minCutPath[-1]])

    return map(lambda x: x - 1, reversed(minCutPath))


def minCutPatch(patch, patchLength, overlap, res, y, x):
    patch = patch.copy()
    dy, dx, _ = patch.shape
    minCut = np.zeros_like(patch, dtype=bool)

    if x > 0:
        left = patch[:, :overlap] - res[y:y + dy, x:x + overlap]
        leftL2 = np.sum(left ** 2, axis=2)
        for i, j in enumerate(minCutPath(leftL2)):
            minCut[i, :j] = True

    if y > 0:
        up = patch[:overlap, :] - res[y:y + overlap, x:x + dx]
        upL2 = np.sum(up ** 2, axis=2)
        for j, i in enumerate(minCutPath(upL2.T)):
            minCut[:i, j] = True

    np.copyto(patch, res[y:y + dy, x:x + dx], where=minCut)

    return patch


def quilt(texture, patchLength, numPatches, mode="cut", sequence=False):
    texture = util.img_as_float(texture)

    overlap = patchLength // 6
    numPatchesHigh, numPatchesWide = numPatches

    h = (numPatchesHigh * patchLength) - (numPatchesHigh - 1) * overlap
    w = (numPatchesWide * patchLength) - (numPatchesWide - 1) * overlap

    res = np.zeros((h, w, texture.shape[2]))

    for i in range(numPatchesHigh):
        for j in range(numPatchesWide):
            y = i * (patchLength - overlap)
            x = j * (patchLength - overlap)

            if i == 0 and j == 0 or mode == "random":
                patch = randomPatch(texture, patchLength)
            elif mode == "best":
                patch = randomBestPatch(texture, patchLength, overlap, res, y, x)
            elif mode == "cut":
                patch = randomBestPatch(texture, patchLength, overlap, res, y, x)
                patch = minCutPatch(patch, patchLength, overlap, res, y, x)

            res[y:y + patchLength, x:x + patchLength] = patch

            if sequence:
                io.imshow(res)
                io.show()

    return res


def quiltSize(texture, patchLength, shape, mode="cut"):
    overlap = patchLength // 6
    h, w = shape

    numPatchesHigh = math.ceil((h - patchLength) / (patchLength - overlap)) + 1 or 1
    numPatchesWide = math.ceil((w - patchLength) / (patchLength - overlap)) + 1 or 1
    res = quilt(texture, patchLength, (numPatchesHigh, numPatchesWide), mode)

    return res[:h, :w]