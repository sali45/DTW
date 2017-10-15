from DTW import dtw
import fastdtw
import numpy as np

x = np.array([1, 2, 3, 4, 5], dtype='float')
y = np.array([1, 2, 3, 4, 5], dtype='float')
window = []
print fastdtw.dtw(x, y)
print dtw(x, y, None, None)


def test_dtw_distance(s1, s2):
    assert(dtw(s1, s2, None, None)[0] == fastdtw.dtw(s1, s2)[0])
    print "Test distance passed"


def test_dtw_path(s1, s2):
    assert(dtw(s1, s2, None, None)[1] == fastdtw.dtw(s1, s2)[1])
    print "Test path passed"

test_dtw_distance(x, y)
test_dtw_path(x, y)

x = np.array([1, 2, 3, 4, 5], dtype='float')
y = np.array([2, 3, 4], dtype='float')

test_dtw_distance(x, y)
test_dtw_path(x, y)

x = np.array([[1, 2, 3, 4, 5]], dtype='float')
y = np.array([[2, 3, 4, 4, 6]], dtype='float')

test_dtw_distance(x, y)
test_dtw_path(x, y)


def sliding_window(sequence, templates, threshold):
    matches = {}
    sliding_window_size = 1 #sec
    step_size = 50 * (sliding_window_size / 4)  # 50 Hz, 25% overlap
    for i in range(len(sequence), step_size):
        segment = sequence[i:i+step_size]
        for template in templates:
            dist = dtw(segment,template)[0]
            if dist < threshold:
                matches[segment] = template
