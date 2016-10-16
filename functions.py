from scipy.ndimage import binary_erosion, binary_dilation
from scipy.misc import imread
import numpy as np
from copy import copy


def read_binary_image(filename):
    image = np.invert(np.array(imread(filename, flatten=True), dtype=np.bool))
    return image


def is_empty_binary_image(image):
    return True if np.count_nonzero(image) == 0 else False


def build_skeleton_binary_image(image, structure):
    n, y1, S = 0, copy(image), []
    while True:
        y2 = binary_erosion(y1, structure)
        if is_empty_binary_image(y2):
            S.append(y1)
            return S
        y3 = binary_dilation(y2, structure)
        S.append(y1 - y3)
        n, y1 = n + 1, y2


def reconstruction_binary_image(S, structure):
    n, image = len(S) - 1, np.zeros(S[0].shape, dtype=np.bool)
    while True:
        image = image + S[n]
        if n == 0:
            return image
        image, n = binary_dilation(image, structure), n - 1


def x(image, A, B):
    return binary_erosion(image, A) * binary_dilation(np.invert(image), B)


def build_convex_hull(image):
    T = {1: np.array([[1, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=np.bool),
         2: np.array([[1, 1, 0], [1, 0, 0], [1, 0, 0]], dtype=np.bool),
         3: np.array([[0, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=np.bool),
         4: np.array([[0, 0, 1], [0, 0, 1], [0, 1, 1]], dtype=np.bool),
         5: np.array([[1, 1, 1], [1, 0, 0], [0, 0, 0]], dtype=np.bool),
         6: np.array([[1, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.bool),
         7: np.array([[0, 0, 0], [0, 0, 1], [1, 1, 1]], dtype=np.bool),
         8: np.array([[0, 1, 1], [0, 0, 1], [0, 0, 1]], dtype=np.bool)}
    B = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.bool)
    def phi(im):
        im = copy(im)
        im = im + x(im, T[1], B)
        im = im + x(im, T[2], B)
        im = im + x(im, T[3], B)
        im = im + x(im, T[4], B)
        im = im + x(im, T[5], B)
        im = im + x(im, T[6], B)
        im = im + x(im, T[7], B)
        im = im + x(im, T[8], B)
        return im
    for _ in range(1000):
        image = phi(image)
    return image
