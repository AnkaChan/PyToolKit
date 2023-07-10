import glob

from PyToolkit import *
from ..Utility.Path import *
import numpy as np

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