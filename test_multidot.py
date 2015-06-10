
import numpy as np
from hunse_tools.timing import tic, toc

m, n = 10, 10
k = 10000

mats = np.random.randn(k, m, n)
vecs = np.random.randn(k, n)

tic()
dots = np.vstack([np.dot(a, v) for a, v in zip(mats, vecs)])
toc()

tic()
dots2 = (mats * vecs[:, None, :]).sum(2)
toc()

# print dots
# print dots2
assert np.allclose(dots, dots2)
