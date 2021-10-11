import numpy as np
import cv2


def angles_between_vecs_batch(v0, v1):
    """ input: v0 and v1 are Nx2 numpy arrays of N 2D vectors (batch)
        output: Nx1 array of angles (unsigned)
    """
    return np.rad2deg(np.arccos(np.clip((v0 * v1).sum(axis=1) / (np.linalg.norm(v0, axis=1) * np.linalg.norm(v1, axis=1)), -1, 1)))

def angles_quad_batch(points):
    """ input: 4 x N x 2 tensor of N 2D quads
        output: N x 4 matrix of internal quad angles (unsigned)
    """
    a0 = angles_between_vecs_batch(points[:,1,:] - points[:,0,:], points[:,3,:] - points[:,0,:])
    a1 = angles_between_vecs_batch(points[:,2,:] - points[:,1,:], points[:,0,:] - points[:,1,:])
    a2 = angles_between_vecs_batch(points[:,1,:] - points[:,2,:], points[:,3,:] - points[:,2,:])
    a3 = angles_between_vecs_batch(points[:,0,:] - points[:,3,:], points[:,2,:] - points[:,3,:])
    return np.array([a0, a1, a2, a3]).transpose()

def edgelens_quad_batch(points):
    """ input: 4 x N x 2 tensor of N 2D quads
        output: N x 4 matrix of quad edges
    """
    e1 = np.linalg.norm(points[:,1,:] - points[:,0,:], axis=1)
    e2 = np.linalg.norm(points[:,2,:] - points[:,1,:], axis=1)
    e3 = np.linalg.norm(points[:,3,:] - points[:,2,:], axis=1)
    e4 = np.linalg.norm(points[:,0,:] - points[:,3,:], axis=1)
    return np.array([e1, e2, e3, e4]).transpose()


def bilinearInterpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1);
    x1 = np.clip(x1, 0, im.shape[1]-1);
    y0 = np.clip(y0, 0, im.shape[0]-1);
    y1 = np.clip(y1, 0, im.shape[0]-1);

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id

def homographyWarpping(img, qv_img, qv_target, outSize, flags=cv2.INTER_NEAREST):
    """ qv_img ... 4x2 pixel coordinates of four quad vertices (clockwise)
        img ... image where the quad lives (greyscale but with RGB)
        returns: 104x104x1 subimage warped to canonical position
    """
    # qv_canonical = np.array([[20, 20], [84, 20], [84, 84], [20, 84]], dtype=np.float64)
    M = cv2.getPerspectiveTransform(qv_img.astype(np.float32), qv_target).astype(np.float32)
    wimg = cv2.warpPerspective(img, M, outSize, flags=flags)
    return wimg