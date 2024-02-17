import numpy as np

def randomlyPickDataIndices(numData, numSelection):
    indices = np.arange(0, numData, dtype=np.int)

    np.random.shuffle(indices)

    indices = indices[:numSelection]

    return indices

def randomPermuteData(imgs, labels, numSelection=None):
    assert imgs.shape[0] ==labels.shape[0], "length of images and labels does not match"
    if numSelection is None:
        randomIndices = np.random.permutation(imgs.shape[0])
    else:
        randomIndices = randomlyPickDataIndices(imgs.shape[0], numSelection)

    imgs = imgs[randomIndices, ...]
    labels = labels[randomIndices, ...]

    return imgs, labels

def toOneHot(labels, numClasses=None):
    if numClasses is None:
        numClasses = np.max(labels) + 1
    oneHone = np.zeros((labels.size, numClasses))
    oneHone[np.arange(labels.size), labels] = 1

    return oneHone