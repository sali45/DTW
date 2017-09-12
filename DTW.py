import numpy as np
from collections import defaultdict

""" Contains a simple Dynamic Time Warping algorithm and unit tests.
"""


def dtw(s1, s2, w=None):
    """Dynamic Time Warping (DTW) implementation
    param s1: first sequence
    param s2: second sequence
    param w: window constraint, i.e. such that |s1[i] - s2[j]| < w
    returns: similarity score and warp path
    """
    cost_mat = defaultdict(lambda: (np.inf,))  # initialize cost matrix as dict where key is cost and values are indices
    #  we keep indices for the path later
    if w is None:  # Just use the entire time series for both series
        w = [(i, j) for i in range(len(s1)) for j in range(len(s2))]
    w = ((i + 1, j + 1) for i, j in w)  # Don't use first row and first col

    cost_mat[0, 0] = (0, 0, 0)  # set first entry to 0 so we can start somewhere.
    for i, j in w:
        cost = np.linalg.norm(s1[i - 1] - s2[j - 1])  # distance between (i, j)-th entry
        cost_mat[i, j] = min((cost_mat[i-1, j][0]+cost, i-1, j),  # insertion
                             (cost_mat[i, j-1][0]+cost, i, j-1),  # deletion
                             (cost_mat[i-1, j-1][0]+cost, i-1, j-1), key=lambda a: a[0])  # match
    path = []
    i, j = len(s1), len(s2)
    while not (i == j == 0):  # while we don't return to the beginning of the warping path
        path.append((i-1, j-1))  # add each element in the warping path starting with the final element in cost_mat
        i, j = cost_mat[i, j][1], cost_mat[i, j][2]  # go up warping path
    path.reverse()  # started from the bottom

    return cost_mat[len(s1), len(s2)][0], path

