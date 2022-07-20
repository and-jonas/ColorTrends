# ======================================================================================================================
# Edge-aware smoothing of the class probability mask using the "fast bilateral solver"
# Code taken from: https://github.com/kuan-wang/The_Bilateral_Solver with minor modifications
# ======================================================================================================================

from pylab import *
import seaborn as sns
from datetime import datetime
# from skimage.io import imread
# from skimage import transform
import os
from scipy.sparse import csr_matrix
from scipy.sparse import diags
from scipy.sparse.linalg import cg
import imageio
import cv2


sns.set_style('white')
sns.set_context('notebook')
rcParams['image.interpolation'] = 'nearest'
rcParams['image.cmap'] = 'CMRmap'
rcParams['figure.facecolor'] = 'w'
RGB_TO_YUV = np.array([
    [ 0.299,     0.587,     0.114],
    [-0.168736, -0.331264,  0.5],
    [ 0.5,      -0.418688, -0.081312]])
YUV_TO_RGB = np.array([
    [1.0,  0.0,      1.402],
    [1.0, -0.34414, -0.71414],
    [1.0,  1.772,    0.0]])
YUV_OFFSET = np.array([0, 128.0, 128.0]).reshape(1, 1, -1)


def rgb2yuv(im):
    return np.tensordot(im, RGB_TO_YUV, ([2], [1])) + YUV_OFFSET


def yuv2rgb(im):
    return np.tensordot(im.astype(float) - YUV_OFFSET, YUV_TO_RGB, ([2], [1]))


MAX_VAL = 255.0


def get_valid_idx(valid, candidates):
    """Find which values are present in a list and where they are located"""
    locs = np.searchsorted(valid, candidates)
    # Handle edge case where the candidate is larger than all valid values
    locs = np.clip(locs, 0, len(valid) - 1)
    # Identify which values are actually present
    valid_idx = np.flatnonzero(valid[locs] == candidates)
    locs = locs[valid_idx]
    return valid_idx, locs


class BilateralGrid(object):
    def __init__(self, im, sigma_spatial=8, sigma_luma=4, sigma_chroma=4):
        im_yuv = rgb2yuv(im)
        # Compute 5-dimensional XYLUV bilateral-space coordinates
        Iy, Ix = np.mgrid[:im.shape[0], :im.shape[1]]
        x_coords = (Ix / sigma_spatial).astype(int)
        y_coords = (Iy / sigma_spatial).astype(int)
        luma_coords = (im_yuv[..., 0] /sigma_luma).astype(int)
        chroma_coords = (im_yuv[..., 1:] / sigma_chroma).astype(int)
        coords = np.dstack((x_coords, y_coords, luma_coords, chroma_coords))
        coords_flat = coords.reshape(-1, coords.shape[-1])
        self.npixels, self.dim = coords_flat.shape
        # Hacky "hash vector" for coordinates,
        # Requires all scaled coordinates be < MAX_VAL
        self.hash_vec = (MAX_VAL**np.arange(self.dim))
        # Construct S and B matrix
        self._compute_factorization(coords_flat)

    def _compute_factorization(self, coords_flat):
        # Hash each coordinate in grid to a unique value
        hashed_coords = self._hash_coords(coords_flat)
        unique_hashes, unique_idx, idx = \
            np.unique(hashed_coords, return_index=True, return_inverse=True)
        # Identify unique set of vertices
        unique_coords = coords_flat[unique_idx]
        self.nvertices = len(unique_coords)
        # Construct sparse splat matrix that maps from pixels to vertices
        self.S = csr_matrix((np.ones(self.npixels), (idx, np.arange(self.npixels))))
        # Construct sparse blur matrices.
        # Note that these represent [1 0 1] blurs, excluding the central element
        self.blurs = []
        for d in range(self.dim):
            blur = 0.0
            for offset in (-1, 1):
                offset_vec = np.zeros((1, self.dim))
                offset_vec[:, d] = offset
                neighbor_hash = self._hash_coords(unique_coords + offset_vec)
                valid_coord, idx = get_valid_idx(unique_hashes, neighbor_hash)
                blur = blur + csr_matrix((np.ones((len(valid_coord),)),
                                          (valid_coord, idx)),
                                         shape=(self.nvertices, self.nvertices))
            self.blurs.append(blur)

    def _hash_coords(self, coord):
        """Hacky function to turn a coordinate into a unique value"""
        # print ("hash_vec:",self.hash_vec)
        return np.dot(coord.reshape(-1, self.dim), self.hash_vec)

    def splat(self, x):
        return self.S.dot(x)

    def slice(self, y):
        return self.S.T.dot(y)

    def blur(self, x):
        """Blur a bilateral-space vector with a 1 2 1 kernel in each dimension"""
        assert x.shape[0] == self.nvertices
        out = 2 * self.dim * x
        for blur in self.blurs:
            out = out + blur.dot(x)
        return out

    def filter(self, x):
        """Apply bilateral filter to an input x"""
        return self.slice(self.blur(self.splat(x))) /  \
               self.slice(self.blur(self.splat(np.ones_like(x))))


def bistochastize(grid, maxiter=10):
    """Compute diagonal matrices to bistochastize a bilateral grid"""
    m = grid.splat(np.ones(grid.npixels))
    n = np.ones(grid.nvertices)
    for i in range(maxiter):
        n = np.sqrt(n * m / grid.blur(n))
    # Correct m to satisfy the assumption of bistochastization regardless
    # of how many iterations have been run.
    m = n * grid.blur(n)
    Dm = diags(m, 0)
    Dn = diags(n, 0)
    return Dn, Dm

class BilateralSolver(object):
    def __init__(self, grid, params):
        self.grid = grid
        self.params = params
        self.Dn, self.Dm = bistochastize(grid)

    def solve(self, x, w):
        # Check that w is a vector or a nx1 matrix
        if w.ndim == 2:
            assert(w.shape[1] == 1)
        elif w.dim == 1:
            w = w.reshape(w.shape[0], 1)
        A_smooth = (self.Dm - self.Dn.dot(self.grid.blur(self.Dn)))
        w_splat = self.grid.splat(w)
        A_data = diags(w_splat[:,0], 0)
        A = self.params["lam"] * A_smooth + A_data
        xw = x * w
        b = self.grid.splat(xw)
        # Use simple Jacobi preconditioner
        A_diag = np.maximum(A.diagonal(), self.params["A_diag_min"])
        M = diags(1 / A_diag, 0)

        # JA inserted
        w_splat[np.where(w_splat == 0)] = 0.00000001
        # end inserted

        # Flat initialization
        y0 = self.grid.splat(xw) / w_splat
        yhat = np.empty_like(y0)
        for d in range(x.shape[-1]):
            yhat[..., d], info = cg(A, b[..., d], x0=y0[..., d], M=M, maxiter=self.params["cg_maxiter"], tol=self.params["cg_tol"])
        xhat = self.grid.slice(yhat)
        return xhat


def smooth_edge_aware(reference, target):
    """
    Smoothes a grey-scale image using the fast bilateral solver, standardizes the output and applies a binary threshold
    :param reference: the original image
    :param target: the probabily map which needs to be filtered
    :return: the result of the bilateral solver and the thresholded binary mask
    """

    # set parameters
    grid_params = {
        'sigma_luma': 4,  # Brightness bandwidth
        'sigma_chroma': 4,  # Color bandwidth
        'sigma_spatial': 48  # Spatial bandwidth
    }
    bs_params = {
        'lam': 128,  # The strength of the smoothness parameter
        'A_diag_min': 1e-5,  # Clamp the diagonal of the A diagonal in the Jacobi preconditioner.
        'cg_tol': 1e-5,  # The tolerance on the convergence in PCG
        'cg_maxiter': 25  # The number of PCG iterations
    }

    # pseudo confidence map (effectively turned off)
    confidence = np.full((reference.shape[0], reference.shape[1]), pow(2, 16) - 1)

    # create grid
    grid = BilateralGrid(reference, **grid_params)

    t = target.reshape(-1, 1).astype(np.double) / (pow(2, 16) - 1)
    c = confidence.reshape(-1, 1).astype(np.double) / (pow(2, 16) - 1)

    tc_filt = grid.filter(t * c)
    c_filt = grid.filter(c)

    output_filter = (tc_filt / c_filt).reshape(reference.shape[:2])
    output_solver = BilateralSolver(grid, bs_params).solve(t, c).reshape(reference.shape[:2])

    # format output
    np.where(output_solver < 0, 0, output_solver)
    output_solver *= 255.0 / output_solver.max()  # Uses 1 division and image.size multiplications
    output_solver = np.uint8(output_solver)

    # threshold filtered map
    thresh = np.where(output_solver <= 175, 0, 1)

    # # Plot
    # fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    # axs[0, 0].imshow(reference)
    # axs[0, 0].set_title('img')
    # axs[1, 0].imshow(target)
    # axs[1, 0].set_title('orig_mask')
    # axs[0, 1].imshow(output_solver)
    # axs[0, 1].set_title('output_solver')
    # axs[1, 1].imshow(out)
    # axs[1, 1].set_title('final mask')
    # plt.show(block=True)

    return output_solver, thresh
