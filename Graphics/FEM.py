import numpy as np

def buildOrthonormalBasisForTriangle(n, b1, b2):
    if n[2] < 0.:
        a = 1.0 / (1.0 - n[2])
        b = n[0] * n[1] * a
        b1[0] = 1.0 - n[0] * n[0] * a
        b1[1] = -b
        b1[2] = n[0]

        b2[0] = b
        b2[1] = n[1] * n[1] * a - 1.0
        b2[2] = -n[1]

    else:
        a = 1.0 / (1.0 + n[2])
        b = -n[0] * n[1] * a
        b1[0] = 1.0 - n[0] * n[0] * a
        b1[1] = b
        b1[2] = -n[0]

        b2[0] = b
        b2[1] = 1.0 - n[1] * n[1] * a
        b2[2] = -n[1]

def projectTriangleTo2D(b1, b2, v1, v2, v3):
    tri2D = np.array([
    [v1.transpose() @ b1, v1.transpose() @ b2],
    [v2.transpose() @ b1, v2.transpose() @ b2],
    [v3.transpose() @ b1, v3.transpose() @ b2],
    ]).squeeze().transpose()
    return tri2D

def computeDeformationGradientTet_np(x1, x2, x3, x4):
    return np.stack([x2-x1, x3 - x1, x4-x1], axis=1)


def computeDeformationGradientTet_torch(x1, x2, x3, x4):
    import torch
    return torch.stack([x2-x1, x3 - x1, x4-x1], dim=1)