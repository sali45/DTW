from DTW import dtw
import fastdtw
import numpy as np

x = np.array([1, 2, 3, 4, 5], dtype='float')
y = np.array([2, 3, 4], dtype='float')
window = []
print fastdtw.dtw(x, y)
print dtw(x, y)


def test_dtw_distance(s1, s2):
    assert(dtw(s1, s2)[0] == fastdtw.dtw(s1, s2)[0])
    print "Test distance passed"


def test_dtw_path(s1, s2):
    assert(dtw(s1, s2)[1] == fastdtw.dtw(s1, s2)[1])
    print "Test path passed"

test_dtw_distance(x, y)
test_dtw_path(x, y)