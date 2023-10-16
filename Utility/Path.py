import os.path
from pathlib import Path
from os.path import join
import glob
import tqdm

def filePart(fPath):
    fp = Path(fPath)
    _, ext = os.path.splitext(fPath)
    return  (fp.parent.absolute()), fp.stem, ext

def globFolderWithExt(folder, ext, sort=False):
    if sort:
        return sorted(glob.glob(join(folder, "*."+ext)))
    else:
        return glob.glob(join(folder, "*."+ext))

def globFolderWithName(folder, name, sort=False):
    if sort:
        return sorted(glob.glob(join(folder, name)))
    else:
        return glob.glob(join(folder, name))

