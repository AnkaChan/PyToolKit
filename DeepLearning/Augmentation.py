import imgaug as ia
import imgaug.augmenters as iaa


def resizeImage(imageBatch, resizeShape, interpolation='nearest'):
    '''
    :param imageBatch: [?, W, H, ] for grayscale or [?, W, H, 3] for RGB image
    :param resizeShape: [W_new, H_new]
    :return:
    '''

    seq = iaa.Sequential([iaa.Resize({"width": resizeShape[0], "height": resizeShape[1], "interpolation":interpolation})])

    return seq(images=imageBatch)