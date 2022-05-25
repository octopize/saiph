
from saiph.reduction.utils.svd import clip
import numpy as np

def test_clip():
    arr = np.array([-1, -0.005, 0, 0.005, 1])
    result = clip(arr, 0.01)
    np.testing.assert_array_equal(result, [-1, -0.01, 0, 0.01, 1])

