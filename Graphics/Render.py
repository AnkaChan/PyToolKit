import glob

from ..Utility.Path import *
import numpy as np
import tqdm
import pyvista as pv
from functools import partial

def renderFolder_trimesh(inFolder, extName='ply', outFolder=None, camera=None, resolution=[1920, 1080]):
    import trimesh

    if outFolder is None:
        outFolder = join(inFolder, 'trimesh_rendering')
    os.makedirs(outFolder, exist_ok=True)

    inFiles = glob.glob(join(inFolder, "*." + extName))

    # print logged messages
    trimesh.util.attach_to_log()
    log = trimesh.util.log

    # load a mesh

    # get a scene object containing the mesh, this is equivalent to:
    # scene = trimesh.scene.Scene(mesh)
    # a 45 degree homogeneous rotation matrix around
    # the Y axis at the scene centroid
    # rotate = trimesh.transformations.rotation_matrix(
    #     angle=np.radians(10.0),
    #     direction=[0, 1, 0],
    #     point=scene.centroid)

    for meshFile in inFiles:
        mesh = trimesh.load(meshFile)
        scene = mesh.scene()

        # trimesh.constants.log.info('Saving image %d', i)

        # rotate the camera view transform
        if camera is None:
            camera, _geometry = scene.graph[scene.camera.name]
        # camera_new = np.dot(rotate, camera_old)

        # apply the new transform
        scene.graph[scene.camera.name] = camera

        p,n,e = filePart(meshFile)

        # saving an image requires an opengl context, so if -nw
        # is passed don't save the image
        try:
            # increment the file name
            file_name = join(outFolder, n + '.jpg')
            # save a render of the object as a png
            png = scene.save_image(resolution=resolution, visible=True)
            # with open(file_name, 'wb') as f:
            #     f.write(png)
            #     f.close()

        except BaseException as E:
            log.debug("unable to save image", str(E))

def renderMesh(points, faces, padFaces=True, resolution=(1920,2080), scalar=None, scalarFor="points"):
    if padFaces:
        facesPadded = []

        for f in faces:
            facesPadded.append([len(f)])
            facesPadded[-1].extend([v[0] for v in f])
        faces = facesPadded

    mesh = pv.PolyData(np.array(points), faces)
    renderPvMesh(mesh, resolution=resolution, scalar=None, scalarFor="points")

def renderPvMesh(mesh, resolution=(1920,2080), scalar=None, scalarFor="points"):

    pltr = pv.Plotter(window_size=resolution)
    # pltr.set_focus([0, 0, 0])
    # pltr.set_position([40, 0, 0])
    pltr.add_mesh(
        mesh,
        scalars=scalar,
        smooth_shading=True,
        specular=1,
        cmap="jet",
        show_scalar_bar=False,
        preference=scalarFor
    )

    # scalarFor "points" or "cells"

    pltr.show()

def renderFolder_pyvista(inFolder, extName='ply', outFolder=None, yUpInput=False, stride = 1, start=0, end=-1, fps=30, addFrameNumber=True,
                         frameNumberPos='upper_left', camera=None, write=True, resolution=(1280,720), scalars=None, filePrefix='',
                         waitForAdjustCamera=False, cycle=False, **kwargs):
    import pyvista as pv
    from pyvista import _vtk

    corner_mappings = {
        'lower_left': _vtk.vtkCornerAnnotation.LowerLeft,
        'lower_right': _vtk.vtkCornerAnnotation.LowerRight,
        'upper_left': _vtk.vtkCornerAnnotation.UpperLeft,
        'upper_right': _vtk.vtkCornerAnnotation.UpperRight,
        'lower_edge': _vtk.vtkCornerAnnotation.LowerEdge,
        'upper_edge': _vtk.vtkCornerAnnotation.UpperEdge,
        'left_edge': _vtk.vtkCornerAnnotation.LeftEdge,
        'right_edge': _vtk.vtkCornerAnnotation.RightEdge,
    }

    if outFolder is None:
        outFolder = join(inFolder, 'pyvista_rendering')
    os.makedirs(outFolder, exist_ok=True)

    inFiles = glob.glob(join(inFolder, filePrefix + "*." + extName))

    mesh = pv.PolyData(inFiles[0])

    # Distances normalized to [0, 2*pi]

    yUpRot = np.array([[1, 0, 0],
              [0, 0, 1],
              [0, 1, 0],
              ])

    if yUpInput:
        mesh.points = (yUpRot @ mesh.points.transpose()).transpose()

    # Make the movie
    pltr = pv.Plotter(window_size=resolution)
    # pltr.set_focus([0, 0, 0])
    # pltr.set_position([40, 0, 0])
    if scalars is not None:
        pltr.add_mesh(
            mesh,
            scalars=scalars[0],
            specular=1,
            cmap="nipy_spectral",
            show_scalar_bar=False,
            **kwargs
        )
    else:
        pltr.add_mesh(
            mesh,
            scalars=None,
            specular=1,
            cmap="nipy_spectral",
            show_scalar_bar=False,
            **kwargs
        )

    if write:
        pltr.open_movie(join(outFolder, 'vis.mp4'), framerate=fps)

    # pltr.show(interactive_update=True, interactive=True, auto_close=False)
    pltr.show(interactive_update=True,  auto_close=False)

    if addFrameNumber:
        p, n, e = filePart(inFiles[0])
        txtHandle = pltr.add_text('Frame: ' + n, position=corner_mappings[frameNumberPos])

    endFrame = end if end >0 else len(inFiles)
    allFramePts = []
    frameNames = []

    class AnimationState:
        def __init__(s):
            s.i = start
            s.start = False

    def startAnimation(state):
        state.start = True
        print('Animation start.')
    state = AnimationState()

    pltr.add_key_event('a', partial(startAnimation, state=state))

    # for iFrame in tqdm.tqdm(range(start, endFrame, stride)):
    while state.i < endFrame:
        if state.start:
            meshFile = inFiles[state.i]

            if scalars is not None:
                pltr.update_scalars(scalars[state.i], render=False)

            meshNew = pv.PolyData(meshFile)
            pointsNew = np.array(meshNew.points)
            pointsNew = (yUpRot @ pointsNew.transpose()).transpose() if yUpInput else meshNew.points
            pltr.update_coordinates(pointsNew, render=True)
            # mesh.points = pointsNew
            if addFrameNumber:
                p, n, e = filePart(meshFile)
                txtHandle.SetText(corner_mappings[frameNumberPos], 'Frame: ' + n)
                if cycle:
                    frameNames.append(n)

            if write:
                pltr.write_frame()
            else:
                pltr.update()

            if cycle:
                allFramePts.append(pointsNew)

            state.i = state.i + stride
        else:
            pltr.update()


    if cycle:
        iFrame = 0
        while not pltr._closed:
            pltr.update_coordinates(allFramePts[iFrame], render=True)

            if addFrameNumber:

                txtHandle.SetText(corner_mappings[frameNumberPos], 'Frame: ' + frameNames[iFrame])

            pltr.update()
            iFrame = iFrame + 1
            if iFrame >= len(allFramePts):
                iFrame = 0

    pltr.show()