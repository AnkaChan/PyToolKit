import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np

import copy

def resizeImage(imageBatch, resizeShape, interpolation='nearest'):
    '''
    :param imageBatch: [?, W, H, ] for grayscale or [?, W, H, 3] for RGB image
    :param resizeShape: [W_new, H_new]
    :return:
    '''

    seq = iaa.Sequential([iaa.Resize({"width": resizeShape[0], "height": resizeShape[1], "interpolation":interpolation})])

    return seq(images=imageBatch)

def augImageAffine(images, labels,  args, repeat=1, suffle=True):

    seq = iaa.Sequential([
        # iaa.ElasticTransformation(alpha=500, sigma=50),
        # iaa.Multiply(augCfg['mul']),
        iaa.Affine(**args)
        # iaa.AddToHueAndSaturation((-10, 10))  # color jitter, only affects the image
    ])

    if repeat > 1:
        augedImgs = []
        augedLabels = []
        for i in range(repeat):
            augedImgs.append(seq(images=images))
            augedLabels.append(labels)

        augedImgs = np.concatenate(augedImgs, axis=0)
        augedLabels = np.concatenate(augedLabels, axis=0)

    else:
        augedImgs = seq(images=images)
        augedLabels = copy.deepcopy(labels)

    if suffle:
        indices = np.random.permutation(augedImgs.shape[0])
        augedImgs = augedImgs[indices]
        augedLabels = augedLabels[indices]

    return augedImgs, augedLabels