from pxr import Usd, UsdGeom, Sdf
import os

def create_usd_mesh(stage, mesh_path, points, face_vertex_counts, face_vertex_indices):
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
    mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)

def export_usd_mesh(usd_file_path, points, face_vertex_counts, face_vertex_indices):
    stage = Usd.Stage.CreateInMemory()
    create_usd_mesh(stage, '/Mesh', points, face_vertex_counts, face_vertex_indices)
    stage.GetRootLayer().Export(usd_file_path)

def read_usd_mesh(usd_file_path):
    stage = Usd.Stage.Open(usd_file_path)
    mesh = UsdGeom.Mesh.Get(stage, '/Mesh')
    points = mesh.GetPointsAttr().Get()
    face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
    face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
    return points, face_vertex_counts, face_vertex_indices

if __name__ == "__main__":
    # Example usage
    usd_file_path = 'mesh.usda'
    points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    face_vertex_counts = [4]
    face_vertex_indices = [0, 1, 2, 3]

    export_usd_mesh(usd_file_path, points, face_vertex_counts, face_vertex_indices)
    read_points, read_face_vertex_counts, read_face_vertex_indices = read_usd_mesh(usd_file_path)

    print("Read points:", read_points)
    print("Read face vertex counts:", read_face_vertex_counts)
    print("Read face vertex indices:", read_face_vertex_indices)