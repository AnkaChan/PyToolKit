import numpy as np

def randomlyPickDataIndices(numData, numSelection):
    indices = np.arange(0, numData, dtype=np.int)

    np.random.shuffle(indices)

    indices = indices[:numSelection]

    return indices