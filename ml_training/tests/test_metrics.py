import math

import numpy as np

from config import LANDMARK_COORD_SHAPE
from training.metrics import avg_dist, mse

landmark_1_0 = np.array([np.array([1, 0]) for _ in range(LANDMARK_COORD_SHAPE[0])])[np.newaxis, :]
landmark_0_1 = np.array([np.array([0, 1]) for _ in range(LANDMARK_COORD_SHAPE[0])])[np.newaxis, :]


def test_avg_dist():
    dist0 = avg_dist(landmark_1_0, landmark_1_0)
    dist1 = avg_dist(landmark_0_1, landmark_0_1)
    dist2 = avg_dist(landmark_1_0, landmark_0_1)

    assert dist0 == 0.
    assert dist1 == 0.
    assert math.isclose(dist2, math.sqrt(2))


def test_mse():
    dist0 = mse(landmark_1_0, landmark_1_0)
    dist1 = mse(landmark_0_1, landmark_0_1)
    dist2 = mse(landmark_1_0, landmark_0_1)
    assert dist0 == 0.
    assert dist1 == 0.
    assert math.isclose(dist2, 1)
