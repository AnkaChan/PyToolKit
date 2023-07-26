from ..Utility.Path import *

import pyvista as pv
import tqdm

def convertObjFolder(inObjFolder, inFileExt='obj', inFilePrefix='', outFileExt='ply', outVtkFolder=None, processInterval=[],
                     addFaces=False, padABeforeName=False, faceMesh=None, ):
    # addFaces = True
    if outVtkFolder is None:
        outVtkFolder = inObjFolder + '/' + outFileExt

    objFiles = glob.glob(inObjFolder + '/' + inFilePrefix + '*.' + inFileExt)

    if faceMesh is not None:
        meshWithFaces = pv.read(faceMesh)
    else:
        meshWithFaces = None

    os.makedirs(outVtkFolder, exist_ok=True)
    if len(processInterval) == 2:
        objFiles = objFiles[processInterval[0]: processInterval[1]]
    for f in tqdm.tqdm(objFiles, desc=inFileExt + " to vtk"):
        fp = Path(f)

        mesh = pv.read(f)
        if addFaces and meshWithFaces is not None:
            mesh.faces = meshWithFaces.faces
        # else:
        #     mesh.faces = np.empty((0,), dtype=np.int32)
        if padABeforeName:
            outName = outVtkFolder + r'\\A' + fp.stem + '.' + outFileExt
        else:
            outName = outVtkFolder + r'\\' + fp.stem + '.' + outFileExt
        mesh.save(outName)