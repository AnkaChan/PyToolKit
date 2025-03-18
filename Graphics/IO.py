from pathlib import Path
from os.path import join
import re

class PLYMesh:
    def __init__(self, filename):
        self.vertices = []
        self.faces = []
        self.parse_ply(filename)

    def parse_ply(self, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()

        header_ended = False
        vertex_count = 0
        face_count = 0
        vertex_section = False
        face_section = False

        for i, line in enumerate(lines):
            line = line.strip()

            if not header_ended:
                if line.startswith("element vertex"):
                    vertex_count = int(line.split()[-1])
                    vertex_section = True
                elif line.startswith("element face"):
                    face_count = int(line.split()[-1])
                    face_section = True
                elif line == "end_header":
                    header_ended = True
                continue

            if header_ended:
                if vertex_count > 0:
                    # Reading vertex lines
                    vertex_data = list(map(float, line.split()))
                    self.vertices.append(vertex_data)
                    vertex_count -= 1
                    if vertex_count == 0:
                        vertex_section = False
                elif face_count > 0 and vertex_section == False:
                    # Reading face lines
                    face_data = list(map(int, re.findall(r'\d+', line)))[
                                1:]  # Skipping the first number which is the face size (e.g., 3 for triangles)
                    self.faces.append(face_data)
                    face_count -= 1
                    if face_count == 0:
                        face_section = False

def write_ply(vertices, faces, out_path):
    """
    Writes a PLY file from given vertices and faces.

    Parameters:
    vertices (numpy.ndarray): Nx3 array of vertex coordinates.
    faces (numpy.ndarray): Mx3 array of face indices.
    out_path (str): Output file path.
    """
    with open(out_path, 'w') as ply_file:
        # PLY Header
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {len(vertices)}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write(f"element face {len(faces)}\n")
        ply_file.write("property list uchar int vertex_indices\n")
        ply_file.write("end_header\n")

        # Write vertices
        for v in vertices:
            ply_file.write(f"{v[0]} {v[1]} {v[2]}\n")

        # Write faces
        for f in faces:
            ply_file.write(f"3 {f[0]} {f[1]} {f[2]}\n")

def readObj(vt_path, idMinus1=True, convertFacesToOnlyPos=False):
    vts = []
    fs = []
    vns = []
    vs = []
    with open(vt_path, 'r') as objFile:
        lines = objFile.readlines()
        for line in lines:
            l = line.split(' ')
            if '' in l:
                l.remove('')
            if l[0] == 'vt':
                assert len(l) == 3
                u = float(l[1])
                v = float(l[2].split('\n')[0])
                vts.append([u, v])
            elif l[0] == 'vn':
                assert len(l) == 4
                vns.append([float(l[1]),  float(l[2]),  float(l[3])])
            elif l[0] == 'v':
                assert len(l) == 4 or len(l) == 7 # 7 means vertex has color
                vs.append([float(l[1]),  float(l[2]),  float(l[3])])
            elif l[0] == 'f':
                fs_curr = []
                for i in range(len(l) - 1):
                    fi = l[i + 1].split('/')
                    fi[-1] = fi[-1].split('\n')[0]
                    # fi = '{}/{}/{}'.format(fi[0], fi[1], fi[2].split('\n')[0])
                    if idMinus1:
                        f = [int(fi[i])-1 for i in range(len(fi))]
                    else:
                        f = [int(fi[i]) for i in range(len(fi))]
                    if convertFacesToOnlyPos:
                        f = f[0]
                    fs_curr.append(f)
                fs.append(fs_curr)
        objFile.close()
    return vs, vns, vts, fs

def writeObj(vs, vns, vts, fs, outFile, withMtl=False, textureFile=None, convertToMM=False, vIdAdd1=True):
    # write new
    with open(outFile, 'w+') as f:
        fp = Path(outFile)
        outMtlFile = join(str(fp.parent), fp.stem + '.mtl')
        if withMtl:
            f.write('mtllib ./' + fp.stem + '.mtl\n')
            with open(outMtlFile, 'w') as fMtl:
                mtlStr = '''newmtl material_0
    Ka 0.200000 0.200000 0.200000
    Kd 1.000000 1.000000 1.000000
    Ks 1.000000 1.000000 1.000000
    Tr 1.000000
    illum 2
    Ns 0.000000
    map_Kd '''
                assert textureFile is not None
                mtlStr += textureFile
                fMtl.write(mtlStr)

        for i, v in enumerate(vs):

            if convertToMM:
                v[0] = 1000 * v[0]
                v[1] = 1000 * v[1]
                v[2] = 1000 * v[2]
            if len(v) == 3:
                f.write('v {:f} {:f} {:f}\n'.format(v[0], v[1], v[2]))
            elif len(v) == 6:
                f.write('v {:f} {:f} {:f} {:f} {:f} {:f}\n'.format(v[0], v[1], v[2], v[3], v[4], v[5]))

        for i, v in enumerate(vns):
            vn = vns[i]
            f.write('vn {:f} {:f} {:f}\n'.format(vn[0], vn[1], vn[2]))

        for vt in vts:
            f.write('vt {:f} {:f}\n'.format(vt[0], vt[1]))

        if withMtl:
            f.write('usemtl material_0\n')
        for iF in range(len(fs)):
            # if facesToPreserve is not None and iF not in facesToPreserve:
            #     continue
            f.write('f')
            if vIdAdd1:
                for fis in fs[iF]:
                    if  isinstance(fis, list):
                        f.write(' {}'.format('/'.join([str(fi+1) for fi in fis])))
                    else:
                        f.write(' {}'.format(fis+1))

            else:
                for fis in fs[iF]:
                    if isinstance(fis, list):
                        f.write(' {}'.format( '/'.join([str(fi) for fi in fis])))
                    else:
                        f.write(' {}'.format(fis))
            f.write('\n')
        f.close()