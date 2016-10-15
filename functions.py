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
