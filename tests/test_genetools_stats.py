import numpy as np

import malid.external


def test_softmax():
    distances_1d = np.array([10, 5, 200]) * -1
    probabilities_1d = malid.external.genetools_stats.softmax(distances_1d)
    assert np.allclose(probabilities_1d, [0.0067, 0.993, 0], atol=1e-03)

    distances_2d = np.array([[10, 5, 200], [7, 26, 100]]) * -1
    probabilities_2d = malid.external.genetools_stats.softmax(distances_2d)
    assert np.allclose(
        probabilities_2d, np.array([[0.0067, 0.993, 0], [1.0, 0, 0]]), atol=1e-03
    )
