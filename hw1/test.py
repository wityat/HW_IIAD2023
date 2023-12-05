import numpy as np

from hw1 import match_timestamps


def test_match_timestamps():
    assert np.array_equal(match_timestamps(np.array([1, 2, 3]), np.array([1, 2, 3])), np.array([0, 1, 2]))
    assert np.array_equal(match_timestamps(np.array([1, 2, 3]), np.array([2, 3, 4])), np.array([0, 0, 1]))
    assert np.array_equal(match_timestamps(np.array([1, 2, 3]), np.array([3, 4, 5])), np.array([0, 0, 0]))
    assert np.array_equal(match_timestamps(np.array([1, 5, 10]), np.array([2, 6, 11])), np.array([0, 1, 2]))
    assert np.array_equal(match_timestamps(np.array([1, 5, 10]), np.array([0, 6, 12])), np.array([0, 1, 2]))
    assert np.array_equal(match_timestamps(np.array([1, 2, 3]), np.array([1.5, 2.5, 3.5])), np.array([0, 1, 2]))


# Edge cases
def test_match_timestamps_edge_cases():
    assert np.array_equal(match_timestamps(np.array([1]), np.array([1])), np.array([0]))
    assert np.array_equal(match_timestamps(np.array([1]), np.array([2])), np.array([0]))
    assert np.array_equal(match_timestamps(np.array([2]), np.array([1])), np.array([0]))
    assert np.array_equal(match_timestamps(np.array([]), np.array([1, 2, 3])), np.array([]))
    assert np.array_equal(match_timestamps(np.array([1, 2, 3]), np.array([])), np.array([]))
