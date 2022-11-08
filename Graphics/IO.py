from pathlib import Path
from os.path import join

def readObj(vt_path, idMinus1=True):
    vts = []
    fs = []
    vns = []
    vs = []
    with open(vt_path, 'r') as objFile:
        lines = objFile.readlines()
        for line in lines:
            l = line.split(' ')
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
                    fs_curr.append(f)
                fs.append(fs_curr)
        objFile.close()
    return vs, vns, vts, fs

def writeObj(vs, vns, vts, fs, outFile, withMtl=False, textureFile=None, convertToMM=False, addOne=False):
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
            for fis in fs[iF]:
                if addOne:
                    f.write(' {}'.format('/'.join([str(fi+1) for fi in fis])))
                else:
                    f.write(' {}'.format( '/'.join([str(fi) for fi in fis])))
            f.write('\n')
        f.close()