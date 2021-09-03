import os.path
from pathlib import Path
from os.path import join
import glob

def filePart(fPath):
    fp = Path(fPath)
    _, ext = os.path.splitext(fPath)
    return  (fp.parent.absolute()), fp.stem, ext

def globFolderWithExt(folder, ext, sort=False):
    if sort:
        sorted(glob.glob(join(folder, "*."+ext)))
    else:
        return glob.glob(join(folder, "*."+ext))

def globFolderWithExtName(folder, name, sort=False):
    if sort:
        sorted(glob.glob(join(folder, name)))
    else:
        return glob.glob(join(folder, name))

