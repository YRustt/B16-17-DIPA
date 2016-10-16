from scipy.ndimage import binary_erosion, binary_dilation, \
    binary_closing, binary_opening, iterate_structure, \
    grey_closing, grey_opening, grey_dilation
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
    return binary_erosion(image, A) * binary_erosion(np.invert(image), B)


def psi(f):
    def run(image, A, B):
        image = copy(image)
        for a, b in zip(A, B):
            image = f(image, a, b)
        return image
    return run


@psi
def thickening(image, A, B):
    image = copy(image)
    return image + x(image, A, B)


@psi
def thinning(image, A, B):
    image = copy(image)
    return image - x(image, A, B)


def iterate(func, image, A, B, iteration=-1):
    iteration = int(iteration)
    if iteration < 1:
        it1 = func(image, A, B)
        it2 = func(it1, A, B)
        while not np.all(it1 == it2):
            it1, it2 = it2, func(it2, A, B)
        return it2
    else:
        image = copy(image)
        for _ in range(iteration):
            image = func(image, A, B)
        return image


def build_convex_hull_binary_image(image, A, B):
    return iterate(thickening, image, A, B)


def calculate_spectrum_binary_image(image, structure, n):
    if n >= 0:
        return np.count_nonzero(binary_opening(image, iterate_structure(structure, n)) -
                                binary_opening(image, iterate_structure(structure, n + 1)))
    else:
        return np.count_nonzero(binary_closing(image, iterate_structure(structure, -n)) -
                                binary_closing(image, iterate_structure(structure, -n - 1)))


def calculate_spectrum_grey_image(image, structure, n):
    def iterate_grey_structure(structure, n):
        radius = structure.shape[0] // 2
        if n > 0:
            buf = copy(structure)
            for _ in range(n - 1):
                C = np.zeros((buf.shape[0] + 2 * radius, buf.shape[0] + 2 * radius))
                C[radius: -radius, radius: -radius] = buf
                buf = grey_dilation(C, structure=structure)
        else:
            buf = np.array([[structure[radius][radius]]])
        return buf
    if n >= 0:
        return np.sum(grey_opening(image, structure=iterate_grey_structure(structure, n)) -
                      grey_opening(image, structure=iterate_grey_structure(structure, n + 1)))
    else:
        return np.sum(grey_closing(image, structure=iterate_grey_structure(structure, -n)) -
                      grey_closing(image, structure=iterate_grey_structure(structure, -n - 1)))
