import pyvista as pv

def visualizeScalarsOnMesh(pvMesh, scalar, scalarName):
    pvMesh.point_data[scalarName] = scalar

    return pvMesh

def visualizeScalarOnMeshFileToFile(inMesh, scalar, scalarName, outMesh):
    mesh = pv.PolyData(inMesh)

    mesh = visualizeScalarsOnMesh(mesh, scalar, scalarName)

    mesh.save(outMesh)

