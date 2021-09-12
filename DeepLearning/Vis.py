import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import itertools

def visualizeImgsAndLabel(imgs, labels, gridW=10, gridH=10, shuffle=True, cmap='gray', figTitle=None, outFile=None, outDPI=300, closeExistingFigures=False):
    if closeExistingFigures:
        matplotlib.pyplot.close("all")
    fig, axs = plt.subplots(gridH, gridW)
    fig.set_size_inches(20*gridW/10, 20*gridH/10)
    padSize = 100

    indices = list(range((len(imgs))))
    if shuffle:
        np.random.shuffle(indices)

    for i, j in itertools.product(range(gridH), range(gridW)):
        imgId = indices[i * gridW + j]
        img = imgs[imgId, ...]
        # pProj = testgtset[i * gridW + j, :]
        axs[i, j].imshow(np.squeeze(img), cmap=cmap)
        axs[i, j].set_title( labels[imgId])
        axs[i, j].axis('off')

    if figTitle is not None:
        fig.suptitle(figTitle)

    if outFile is not None:
        fig.savefig(outFile, dpi=outDPI, transparent=True, bbox_inches='tight', pad_inches=0)
