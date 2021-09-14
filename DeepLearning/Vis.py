import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import itertools

def visualizeImgsAndLabel(imgs, labels, gridW=10, gridH=10, shuffle=True, cmap='gray', figTitle=None, outFile=None, outDPI=300, closeExistingFigures=False):
    if closeExistingFigures:
        plt.close("all")
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
        if labels is not None:
            axs[i, j].set_title( labels[imgId])
        axs[i, j].axis('off')

    if figTitle is not None:
        fig.suptitle(figTitle)

    if outFile is not None:
        fig.savefig(outFile, dpi=outDPI, transparent=True, bbox_inches='tight', pad_inches=0)

def visualizeImgsSortedByLabel(imgs, labels, gridW=10, gridH=10, randomlyPickLabels=True, shuffle=True, cmap='gray', figTitle=None, outFile=None, outDPI=300, closeExistingFigures=False):
    import random
    if closeExistingFigures:
        plt.close("all")
    fig, axs = plt.subplots(gridH, gridW)
    fig.set_size_inches(20*gridW/10, 20*gridH/10)
    padSize = 100

    allLabelSet = set(labels)
    allLabels = list(allLabelSet)

    labelsTdDataId = {label:[] for iLabel, label in enumerate(allLabels)}

    if randomlyPickLabels:
        random.shuffle(allLabels)

    labelsToShow = allLabels[:gridH]

    for i, label in enumerate(labels):
        labelsTdDataId[label].append(i)

    if shuffle:
        for label in allLabelSet:
            random.shuffle(labelsTdDataId[label])

    for i, j in itertools.product(range(gridH), range(gridW)):
        label = labelsToShow[i]

        if j < len(labelsTdDataId[label]):
            imgId = labelsTdDataId[label][j]
            img = imgs[imgId, ...]
            # pProj = testgtset[i * gridW + j, :]
            axs[i, j].imshow(np.squeeze(img), cmap=cmap)
            if labels is not None:
                axs[i, j].set_title( labels[imgId])
        axs[i, j].axis('off')

    if figTitle is not None:
        fig.suptitle(figTitle)

    if outFile is not None:
        fig.savefig(outFile, dpi=outDPI, transparent=True, bbox_inches='tight', pad_inches=0)

def drawErrCurves(x, y, saveFile=None, title='', xlabel='Epoch', ylabel='Error', closeExistingFigures=False):
    if closeExistingFigures:
        plt.close("all")

    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set(xlabel=xlabel, ylabel=ylabel,
           title=title)
    ax.grid()

    if saveFile is not None:
        fig.savefig(saveFile)
