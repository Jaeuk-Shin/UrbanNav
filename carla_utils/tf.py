import numpy as np


# for UE -> standard (right-handed) coordinate system conversion
# See https://carla.readthedocs.io/en/latest/coordinates/
UE = np.array([
    [0., 1., 0.,  0.],
    [0., 0., -1., 0.],
    [1., 0., 0.,  0.],
    [0., 0., 0.,  1.]
])