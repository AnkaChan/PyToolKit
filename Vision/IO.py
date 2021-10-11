import OpenEXR
import Imath, array
import numpy as np

def readEXR(inFile, toUint8=True, withDepth=True, turnToSRGB=True):
    file = OpenEXR.InputFile(inFile)

    # Compute the size
    dw = file.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # # Read the three color channels as 32-bit floats
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    (R, G, B, Z) = [np.frombuffer(file.channel(Chan, FLOAT), dtype=np.float32) for Chan in ("R", "G", "B", "Z")]
    R=R.reshape(sz[1], sz[0])
    G=G.reshape(sz[1], sz[0])
    B=B.reshape(sz[1], sz[0])
    img = np.stack([R, G, B], axis=-1)

    if turnToSRGB:
        img = np.clip(img, 0, 1)
        mask = img <= 0.0031308
        img[mask] = img[mask]* 12.92
        mask = np.logical_not(mask)
        img[mask] = 1.055 *np.power(img[mask], 1.0 / 2.4) - 0.055

    if toUint8:
        img = (255* img).astype(np.uint8)

    if withDepth:
        depth = Z
        depth = depth.reshape(sz[1], sz[0])
        return img, depth
    else:
        return img
