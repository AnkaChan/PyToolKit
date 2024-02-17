import pyvista as pv

def padFaceListToPV(faces):
    facesPadded = []

    for f in faces:
        facesPadded.append([len(f)])
        facesPadded[-1].extend([v[0] for v in f])

    return facesPadded

def PVFaceListTo2DFaceList(pvFaces, padWithPointNumber=False):
    faceList2D = []

    facePNum = pvFaces[0]
    faceList2D.append([])
    if padWithPointNumber:
        faceList2D[-1].append(facePNum)
    pointCount = 0
    for i in range(1, len(pvFaces)):
        if pointCount < facePNum:
            faceList2D[-1].append(pvFaces[i])
            pointCount += 1
        else:
            pointCount = 0
            facePNum = pvFaces[i]
            faceList2D.append([])
            if padWithPointNumber:
                faceList2D[-1].append(facePNum)

    return faceList2D