from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Kernel import Triangle_3
from CGAL.CGAL_Kernel import Ray_3
from CGAL.CGAL_AABB_tree import AABB_tree_Triangle_3_soup
from numpy import cross, sum, isscalar, spacing, vstack
from numpy.core.umath_tests import inner1d
import numpy as np
import tqdm

def barycentric_coordinates_of_projection(p, q, u, v):
    """Given a point, gives projected coords of that point to a triangle
    in barycentric coordinates.
    See
        **Heidrich**, Computing the Barycentric Coordinates of a Projected Point, JGT 05
        at http://www.cs.ubc.ca/~heidrich/Papers/JGT.05.pdf
    :param p: point to project
    :param q: a vertex of the triangle to project into
    :param u,v: edges of the the triangle such that it has vertices ``q``, ``q+u``, ``q+v``
    :returns: barycentric coordinates of ``p``'s projection in triangle defined by ``q``, ``u``, ``v``
            vectorized so ``p``, ``q``, ``u``, ``v`` can all be ``3xN``
    """

    p = p.T
    q = q.T
    u = u.T
    v = v.T

    n = cross(u, v, axis=0)
    s = sum(n * n, axis=0)

    # If the triangle edges are collinear, cross-product is zero,
    # which makes "s" 0, which gives us divide by zero. So we
    # make the arbitrary choice to set s to epsv (=numpy.spacing(1)),
    # the closest thing to zero
    if isscalar(s):
        s = s if s else spacing(1)
    else:
        s[s == 0] = spacing(1)

    oneOver4ASquared = 1.0 / s
    w = p - q
    b2 = sum(cross(u, w, axis=0) * n, axis=0) * oneOver4ASquared
    b1 = sum(cross(w, v, axis=0) * n, axis=0) * oneOver4ASquared
    b = vstack((1 - b1 - b2, b1, b2))

    return b.T

def pointsToTriangles(points,triangles):
    '''
    find closest point on a give series of triangles to a given point
    :param points:
    :param triangles:
    :return:
    '''

    with np.errstate(all='ignore'):

        # Unpack triangle points
        p0,p1,p2 = np.asarray(triangles).swapaxes(0,1)

        # Calculate triangle edges
        e0 = p1-p0
        e1 = p2-p0
        a = inner1d(e0,e0)
        b = inner1d(e0,e1)
        c = inner1d(e1,e1)

        # Calculate determinant and denominator
        det = a*c - b*b
        invDet = 1. / det
        denom = a-2*b+c

        # Project to the edges
        p  = p0-points[:,np.newaxis]
        d = inner1d(e0,p)
        e = inner1d(e1,p)
        u = b*e - c*d
        v = b*d - a*e

        # Calculate numerators
        bd = b+d
        ce = c+e
        numer0 = (ce - bd) / denom
        numer1 = (c+e-b-d) / denom
        da = -d/a
        ec = -e/c


        # Vectorize test conditions
        m0 = u + v < det
        m1 = u < 0
        m2 = v < 0
        m3 = d < 0
        m4 = (a+d > b+e)
        m5 = ce > bd

        t0 =  m0 &  m1 &  m2 &  m3
        t1 =  m0 &  m1 &  m2 & ~m3
        t2 =  m0 &  m1 & ~m2
        t3 =  m0 & ~m1 &  m2
        t4 =  m0 & ~m1 & ~m2
        t5 = ~m0 &  m1 &  m5
        t6 = ~m0 &  m1 & ~m5
        t7 = ~m0 &  m2 &  m4
        t8 = ~m0 &  m2 & ~m4
        t9 = ~m0 & ~m1 & ~m2

        u = np.where(t0, np.clip(da, 0, 1), u)
        v = np.where(t0, 0, v)
        u = np.where(t1, 0, u)
        v = np.where(t1, 0, v)
        u = np.where(t2, 0, u)
        v = np.where(t2, np.clip(ec, 0, 1), v)
        u = np.where(t3, np.clip(da, 0, 1), u)
        v = np.where(t3, 0, v)
        u *= np.where(t4, invDet, 1)
        v *= np.where(t4, invDet, 1)
        u = np.where(t5, np.clip(numer0, 0, 1), u)
        v = np.where(t5, 1 - u, v)
        u = np.where(t6, 0, u)
        v = np.where(t6, 1, v)
        u = np.where(t7, np.clip(numer1, 0, 1), u)
        v = np.where(t7, 1-u, v)
        u = np.where(t8, 1, u)
        v = np.where(t8, 0, v)
        u = np.where(t9, np.clip(numer1, 0, 1), u)
        v = np.where(t9, 1-u, v)


        # Return closest points
        return (p0.T +  u[:, np.newaxis] * e0.T + v[:, np.newaxis] * e1.T).swapaxes(2,1)

def searchForClosestPoints(sourceVs, targetVs):
    closestPts = []

    for sv in sourceVs:
        minDst = 100000
        closestP = None
        # for tv in targetVs:
        #     dst = np.linalg.norm(sv - tv)
        #     if dst < minDst:
        #         minDst = dst
        #         closestP = tv

        dists = np.sum(np.square(targetVs - sv), axis=1)
        tvId = np.argmin(dists)

        closestPts.append(targetVs[tvId, :])
    return np.array(closestPts)

def searchForClosestPoints(sourceVs, targetVs, tree):
    closestPts = []
    dis = []
    for sv in sourceVs:

        d, tvId = tree.query(sv)
        closestPts.append(targetVs[tvId, :])
        dis.append(d)
    return np.array(closestPts), np.array(dis)

def toP(vNp):
    vNp = vNp.astype(np.float64)
    return Point_3(vNp[0], vNp[1], vNp[2])

def fromP(p3):
    return np.array([p3.x(), p3.y(), p3.z()])

def searchForClosestPointsOnTriangle(sourceVs, targetVs, targetFs):
    # triangles = np.array([[targetVs[f[0], :], targetVs[f[1], :], targetVs[f[2], :]] for f in targetFs])
    triangles = [Triangle_3(toP(targetVs[f[0], :]), toP(targetVs[f[1], :]), toP(targetVs[f[2], :])) for f in targetFs]
    tree = AABB_tree_Triangle_3_soup(triangles)
    closestPts = []
    for i, v in enumerate(sourceVs):
        #print("query for vertex: ", i)
        closestPts.append(fromP(tree.closest_point(toP(v))))
    return np.array(closestPts)

def searchForClosestPointsOnTriangleWithBarycentric(sourceVs, targetVs, targetFs, returnDis=False):
    # triangles = np.array([[targetVs[f[0], :], targetVs[f[1], :], targetVs[f[2], :]] for f in targetFs])
    triangles = [Triangle_3(toP(targetVs[f[0], :]), toP(targetVs[f[1], :]), toP(targetVs[f[2], :])) for f in targetFs]
    tree = AABB_tree_Triangle_3_soup(triangles)
    closestPts = []
    trianglesId = []


    for i, v in tqdm.tqdm(enumerate(sourceVs)):
        #print("query for vertex: ", i)
        # closestPts.append(fromP(tree.closest_point(toP(v))))
        p, id = tree.closest_point_and_primitive(toP(v))

        closestPts.append(fromP(p))
        trianglesId.append(id)

    barycentrics = []
    for tId, p in zip(trianglesId, closestPts):
        a = p
        t = targetFs[tId, :]
        tp1 = targetVs[t[0], :]
        u = targetVs[t[1], :] - tp1
        v = targetVs[t[2], :] - tp1
        c = barycentric_coordinates_of_projection(a, tp1, u, v)
        assert np.min(c) > -0.0001
        assert np.sum(c) >= 0.99999

        barycentrics.append(c[0, :])

    if returnDis:
        closestPts = np.array(closestPts)
        diff = sourceVs - closestPts
        dis = np.sqrt(diff[:, 0]**2 + diff[:, 0]**2 + diff[:, 0]**3)

        # closestPts, barycentrics, trianglesId, dis
        return closestPts, np.array(barycentrics), np.array(trianglesId), dis
    else:
        return np.array(closestPts), np.array(barycentrics), np.array(trianglesId)