import pyvista as pv
import numpy as np

def visualizeScalarsOnMesh(pvMesh, scalar, scalarName):
    pvMesh.point_data[scalarName] = scalar

    return pvMesh

def visualizeScalarOnMeshFileToFile(inMesh, scalar, scalarName, outMesh):
    mesh = pv.PolyData(inMesh)

    mesh = visualizeScalarsOnMesh(mesh, scalar, scalarName)

    mesh.save(outMesh)

def drawCorrs(pts1, pts2, outCorrFile):
    import vtk
    ptsVtk = vtk.vtkPoints()
    ptsAll = np.vstack([pts1, pts2])
    numPts = pts1.shape[0]

    # assert pts1.shape[0] == pts2.shape[0]

    # pts.InsertNextPoint(p1)
    for i in range(ptsAll.shape[0]):
        ptsVtk.InsertNextPoint(ptsAll[i, :].tolist())

    polyData = vtk.vtkPolyData()
    polyData.SetPoints(ptsVtk)

    lines = vtk.vtkCellArray()

    for i in range(pts1.shape[0]):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, i)  # the second 0 is the index of the Origin in the vtkPoints
        line.GetPointIds().SetId(1, i + numPts)  # the second 1 is the index of P0 in the vtkPoints
        # line.
        lines.InsertNextCell(line)

    polyData.SetLines(lines)

    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(polyData)
    writer.SetFileName(outCorrFile)
    writer.Update()