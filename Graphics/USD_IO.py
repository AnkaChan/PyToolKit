from pxr import Usd, UsdGeom, Sdf

def create_usd_mesh(stage, mesh_path, points, face_vertex_counts, face_vertex_indices):
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
    mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)

def export_usd_mesh(usd_file_path, points, face_vertex_counts, face_vertex_indices, path='/Mesh'):
    stage = Usd.Stage.CreateInMemory()
    create_usd_mesh(stage, path, points, face_vertex_counts, face_vertex_indices)
    stage.GetRootLayer().Export(usd_file_path)

def read_usd_mesh(usd_file_path, path='/Mesh'):
    stage = Usd.Stage.Open(usd_file_path)
    mesh = UsdGeom.Mesh.Get(stage, path)
    points = mesh.GetPointsAttr().Get()
    face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
    face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
    return points, face_vertex_counts, face_vertex_indices

if __name__ == "__main__":
    # Example usage
    # in_file = r'D:\Code\Graphics\warp - dev - demos\Data\BVHQuery\Debug\frame.obj'
    # read
    #
    # export_usd_mesh(usd_file_path, points, face_vertex_counts, face_vertex_indices)
    # read_points, read_face_vertex_counts, read_face_vertex_indices = read_usd_mesh(usd_file_path)
    #
    # print("Read points:", read_points)
    # print("Read face vertex counts:", read_face_vertex_counts)
    # print("Read face vertex indices:", read_face_vertex_indices)
    pass