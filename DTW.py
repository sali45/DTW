import numpy as np

""" Contains a simple Dynamic Time Warping algorithm and unit tests.
TODO: Add lower bound constraint such as lb_keogh
"""


def dtw_distance(s1, s2, w):
    """Dynamic Time Warping (DTW) implementation
    param s1: first sequence
    param s2: second sequence
    param w: window constraint, i.e. such that |s1[i] - s2[j]| < w
    returns: similarity score, cost matrix, and warp path
    """
    cost_mat = np.empty((len(s1), len(s2)))  # initialize cost matrix
    cost_mat.fill(float("inf"))  # initialize whole matrix with dummy values of infinity

    w = max(w, abs(len(s1) - len(s2)))  # w can't be greater than difference between the lengths of the two sequences

    cost_mat[0, 0] = 0  # set first entry to 0 so we can start somewhere.

    for i in range(1, len(s1)):
        for j in range(max(1, i - w), min(len(s2), i + w)):  # apply window constraint
            cost = s1[i] - s2[j]  # distance between (i, j)-th entry
            cost_mat[i, j] = cost + min(cost_mat[i - 1, j],  # insertion
                                        cost_mat[i, j - 1],  # deletion
                                        cost_mat[i - 1, j - 1])   # match

    return cost_mat[len(s1) - 1, len(s2) - 1]  # why do I return the last element?
