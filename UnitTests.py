from DTW import dtw_distance

x = [0.0, 0.0, 1.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 0.0]
y = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 2.0, 0.0]


def test_dtw_distance(s1, s2, w):
    assert(dtw_distance(s1, s2, w) == -19)
    print "Test 1 passed"

test_dtw_distance(x, y, 4)
