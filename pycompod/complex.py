"""
complex.py
----------

Cell complex from planar primitive arrangement.

A linear cell complex is constructed from planar primitives
with adaptive binary space partitioning: upon insertion of a primitive
only the local cells that are intersecting it will be updated,
so will be the corresponding adjacency graph of the complex.
"""

import time, multiprocessing, pickle, logging, trimesh, copy, warnings, sys, os
from pathlib import Path

import numpy as np
from tqdm import trange, tqdm
import networkx as nx
from sage.all import QQ, RDF, ZZ, Polyhedron, vector, arctan2
from treelib import Tree
from collections import defaultdict
from shapely.geometry import Polygon
from shapely import contains_xy
from multiprocessing import Process

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import gco # pip install gco-wrapper

from .export_complex import PolyhedralComplexExporter
from .logger import make_logger

from .plane import ProjectedConvexHull, PyPlane
from .export_plane import PlaneExporter

class PolyhedralComplex:
    """
    Class of cell complex from planar primitive arrangement.
    """
    def __init__(self, vertex_group, padding=0.02, insertion_threshold=0, device='cpu', debug_export=False, verbosity=logging.WARN):
        """
        Init PolyhedralComplex.
        Class of polyhedral complex from planar primitive arrangement.

        Parameters
        ----------
        vertex_group: The input primitives.
        padding: Padding of bounding box of primitives.
        """

        # set random seed to have deterministic results for point sampling and filling of convex hull point arrays.
        np.random.seed(42)

        self.planeExporter = PlaneExporter()
        self.complexExporter = PolyhedralComplexExporter(self)

        self.vg = vertex_group
        # basically just a name change, to make sure that the negative indices for the boundary planes are never used on this array and not on the split_planes array
        self.vg.input_planes = copy.deepcopy(self.vg.planes)
        self.n_points = np.concatenate(self.vg.groups).shape[0]
        del self.vg.planes

        self.verbosity = verbosity
        self.logger = make_logger(name="COMPOD",level=verbosity)

        self.logger.debug('Init cell complex with padding {}'.format(padding))

        self.cells = dict()
        self.tree = None
        self.graph = None
        self.device = device
        if self.device == 'gpu':
            import torch
            self.torch = torch
        else:
            self.torch = None

        self.partition_initialized = False
        self.polygons_initialized = False
        self.polygons_constructed = False
        self.partition_labelled = False
        self.tree_simplified = False

        # init the bounding box
        self.padding = padding
        self.bounding_poly = self._init_bounding_box(padding=self.padding)

        self.insertion_threshold = insertion_threshold
        self.tree_mode = Tree.DEPTH

        self.debug_export = debug_export
        if self.debug_export:
            self.logger.warning('Debug export activated. Turn off for faster processing.')





    def _init_bounding_box(self,padding):

        self.bounding_verts = []

        pmin = vector([QQ(self.vg.points[:,0].min()),QQ(self.vg.points[:,1].min()),QQ(self.vg.points[:,2].min())])
        pmax = vector([QQ(self.vg.points[:,0].max()),QQ(self.vg.points[:,1].max()),QQ(self.vg.points[:,2].max())])

        d = pmax-pmin
        d = d*QQ(padding)
        pmin = pmin-d
        pmax = pmax+d


        self.bounding_verts.append(pmin)
        self.bounding_verts.append([pmin[0],pmax[1],pmin[2]])
        self.bounding_verts.append([pmin[0],pmin[1],pmax[2]])
        self.bounding_verts.append([pmin[0],pmax[1],pmax[2]])
        self.bounding_verts.append(pmax)
        self.bounding_verts.append([pmax[0],pmin[1],pmax[2]])
        self.bounding_verts.append([pmax[0],pmax[1],pmin[2]])
        self.bounding_verts.append([pmax[0],pmin[1],pmin[2]])

        return Polyhedron(vertices=self.bounding_verts)


    def _inequalities(self, plane):
        """
        Inequalities from plane parameters.

        Parameters
        ----------
        plane: (4,) float
            Plane parameters

        Returns
        -------
        positive: (4,) float
            Inequality of the positive half-plane
        negative: (4,) float
            Inequality of the negative half-plane
        """
        positive = [QQ(plane[-1]), QQ(plane[0]), QQ(plane[1]), QQ(plane[2])]
        negative = [QQ(-element) for element in positive]
        return positive, negative


    def add_bounding_box_planes(self):

        self.logger.info("Add bounding box planes...")
        self.logger.debug("Bounding box planes will be appended to self.vg.input_planes")

        pmin = vector(self.bounding_poly.bounding_box()[0])
        pmax = vector(self.bounding_poly.bounding_box()[1])
        d = pmax-pmin
        d = d*QQ(self.padding*10)
        pmin = pmin-d
        pmax = pmax+d

        bounding_verts = []
        bounding_verts.append(pmin)
        bounding_verts.append([pmin[0],pmax[1],pmin[2]])
        bounding_verts.append([pmin[0],pmin[1],pmax[2]])
        bounding_verts.append([pmin[0],pmax[1],pmax[2]])
        bounding_verts.append(pmax)
        bounding_verts.append([pmax[0],pmin[1],pmax[2]])
        bounding_verts.append([pmax[0],pmax[1],pmin[2]])
        bounding_verts.append([pmax[0],pmin[1],pmin[2]])

        outside_poly = Polyhedron(vertices=bounding_verts)

        bb_verts = self.bounding_poly.bounding_box()
        bb_planes = []
        bb_planes.append([-1,0,0,bb_verts[0][0]])
        bb_planes.append([1,0,0,-bb_verts[1][0]])
        bb_planes.append([0,-1,0,bb_verts[0][1]])
        bb_planes.append([0,1,0,-bb_verts[1][1]])
        bb_planes.append([0,0,-1,bb_verts[0][2]])
        bb_planes.append([0,0,1,-bb_verts[1][2]])
        bb_planes = np.array(bb_planes,dtype=object)

        dtype = self.vg.input_planes.dtype
        self.vg.input_planes = np.vstack((self.vg.input_planes,np.flip(bb_planes,axis=0).astype(dtype)))
        bb_color = np.zeros(shape=(6,3))+[255,0,255]
        self.vg.plane_colors = np.vstack((self.vg.plane_colors,bb_color.astype(self.vg.plane_colors.dtype)))

        max_node_id = max(list(self.graph.nodes))
        for i,plane in enumerate(bb_planes):


            hspace_neg, hspace_pos = self._inequalities(plane)
            op = outside_poly.intersection(Polyhedron(ieqs=[hspace_neg]))

            if self.debug_export:
                self.complexExporter.write_cell(os.path.join(self.debug_export,"bounding_box_cells"),op,count=-(i+1))

            self.cells[i+1+max_node_id] = op
            self.graph.add_node(i+1+max_node_id, bounding_box=True)

            for cell_id in list(self.graph.nodes):

                intersection = op.intersection(self.cells.get(cell_id))

                if intersection.dim() == 2:
                    self.graph.add_edge(i+1+max_node_id,cell_id, intersection=None, vertices=[],
                                   supporting_plane_id=-(i+1), convex_intersection=False, bounding_box=True)



    def _sort_vertex_indices_by_angle_exact(self,points,plane):
        '''order vertices of a convex polygon:
        https://blogs.sas.com/content/iml/2021/11/17/order-vertices-convex-polygon.html#:~:text=Order%20vertices%20of%20a%20convex%20polygon&text=You%20can%20use%20the%20centroid,vertices%20of%20the%20convex%20polygon
        '''
        ## project to plane
        # pp=PyPlane(plane).project_points_to_plane_coordinate_system(points)

        max_coord=PyPlane(plane).max_coord
        pp = np.delete(points,max_coord,axis=1)

        center = np.mean(pp,axis=0)
        vectors = pp - center
        # vectors = vectors/np.linalg.norm(vectors)
        radians = []
        for v in vectors:
            radians.append(float(arctan2(v[1],v[0])))
        radians = np.array(radians)
        # if (np.unique(radians,return_counts=True)[1]>1).any():
        #     print("WARNING: same exact radians in _sort_vertex_indices_by_angle_exact")

        return np.argsort(radians)

    def _sorted_vertex_indices(self,adjacency_matrix):
        """
        Return sorted vertex indices.

        Parameters
        ----------
        adjacency_matrix: matrix
            Adjacency matrix

        Returns
        -------
        sorted_: list of int
            Sorted vertex indices
        """

        pointer = 0
        sorted_ = [pointer]
        for _ in range(len(adjacency_matrix[0]) - 1):
            connected = np.where(adjacency_matrix[pointer])[0]  # two elements
            if connected[0] not in sorted_:
                pointer = connected[0]
                sorted_.append(connected[0])
            else:
                pointer = connected[1]
                sorted_.append(connected[1])
        return sorted_

    def _orient_polygon_exact(self, points, outside, return_orientation = False):
        # check for left or right orientation
        # https://math.stackexchange.com/questions/2675132/how-do-i-determine-whether-the-orientation-of-a-basis-is-positive-or-negative-us

        i = 0
        cross=0
        while np.sum(cross*cross) == 0:
            try:
                a = vector(points[i+1] - points[i])
                # a = a/a.norm()
                b = vector(points[i+2] - points[i])
                # b = b/b.norm()
                cross = a.cross_product(b)
                i+=1
            except:
                return 0


        c = vector(np.array(outside,dtype=object) - points[i])
        # c = c/c.norm()
        # cross = cross/cross.norm()
        dot = cross.dot_product(c)

        if return_orientation:
            return dot < 0, cross
        else:
            return dot < 0


    def _get_intersection(self, e0, e1):

        if "vertices" in self.graph[e0][e1] and len(self.graph[e0][e1]["vertices"]):
            pts = []
            for v in self.graph[e0][e1]["vertices"]:
                pts.append(tuple(v))
            pts = list(set(pts))
            intersection_points = np.array(pts, dtype=object)
        elif "intersection" in self.graph[e0][e1] and self.graph[e0][e1] is not None:
            self.logger.warning("This intersection should have been computed before!")
            intersection_points = np.array(self.graph[e0][e1]["intersection"].vertices_list(), dtype=object)
        else:
            self.logger.warning("This intersection should have been computed before!")
            intersection = self.cells.get(e0).intersection(self.cells.get(e1))
            assert(intersection.dim()==2)
            intersection_points = np.array(intersection.vertices_list(), dtype=object)

        return intersection_points


    def save_colored_soup(self, out_file):

        """
        Extracts a polygon soup from the labelled polyhedral complex. Polygons are colored according to the planar region they belong to.

        :param out_file: File to store the soup (has to be a .ply file).

        """

        self.logger.info('Save colored surface soup...')

        fcolors = []
        pcolors = []
        all_points = []
        # for cgal export
        faces = []
        n_points = 0
        for e0, e1 in self.graph.edges:

            c0 = self.graph.nodes[e0]
            c1 = self.graph.nodes[e1]

            if c0["occupancy"] != c1["occupancy"]:

                plane_id = self.graph.edges[e0, e1]["supporting_plane_id"]
                # col = self.vg.plane_colors[plane_id] if plane_id > -1 else np.array([50,50,50])
                col = self.vg.plane_colors[plane_id]
                fcolors.append(col)

                intersection_points = self._get_intersection(e0, e1)

                plane_id = self.graph[e0][e1]["supporting_plane_id"]
                plane = self.vg.input_planes[plane_id]
                correct_order = self._sort_vertex_indices_by_angle_exact(intersection_points,plane)

                assert (len(intersection_points) == len(correct_order))
                intersection_points = intersection_points[correct_order]

                if (len(intersection_points) < 3):
                    print("ERROR: Encountered facet with less than 3 vertices.")
                    sys.exit(1)

                ## orient polygon
                outside = self.cells.get(e0).center() if c1["occupancy"] else self.cells.get(e1).center()
                if self._orient_polygon_exact(intersection_points, outside):
                    intersection_points = np.flip(intersection_points, axis=0)

                for pt in intersection_points:
                    all_points.append(pt)
                    pcolors.append(col)

                faces.append(np.arange(len(intersection_points)) + n_points)
                n_points += len(intersection_points)

        all_points = np.array(all_points, dtype=float)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        self.logger.debug('Save colored polygon soup to {}'.format(out_file))

        self.complexExporter.write_colored_soup_to_ply(out_file, points=all_points, facets=faces, pcolors=pcolors, fcolors=fcolors)


    def save_simplified_surface(self, out_file, triangulate = False, simplify_edges = True, backend = "python"):

        """
        Extracts a watertight simplified surface mesh from the labelled polyhedral complex. Each planar region of the
        mesh is either triangulated (if it contains holes) or represented as one (region with one connected component)
        or several (region with multiple connected components) polygons.

        :param out_file: File to store the mesh.

        :param triangulate: Flag that controls if all regions should be triangulated or not. Necessary for correct orientation.

        :param simplify_edges: Flag that controls if region boundaries should only contain corner vertices or all vertices of the decomposition.
        """

        try:
            from pycompose import pdse
        except:
            self.logger.error(
                "Could not import pdse. Please install COMPOSE from https://github.com/raphaelsulzer/compod#compose.")
            raise ModuleNotFoundError

        def _get_region_borders(region):

            """
            Get all border edges of region.

            :param region: Tuple of region id and region polygons.
            :return: All unsorted edges of a region.
            """

            rid = region[0]
            region = region[1]

            region_facets = []
            region_edges = []
            for pid in region:
                face = facets[pid]
                region_facets.append(face)

                nf = len(face)
                for i in range(nf):
                    region_edges.append([face[i%nf],face[(i+1)%nf]])

            # if self.debug_export:
            #     of = out_file[:-4] + str(rid) + ".off"
            #     self.complexExporter.write_surface_to_off(of, points=points, facets=region_facets)

            # all edges that only appear once per region are border edges of that region
            unique, inverse, count = np.unique(np.sort(region_edges),return_inverse=True,return_counts=True,axis=0)
            return np.array(region_edges)[(count==1)[inverse]]

        def _orient_facets(facets, region):
            """
            Orient facets by checking if the current orientation is the same as the one of the region.
            :param facets:
            :return:
            """

            # a = points[facets[0][0]]
            # b = points[facets[0][1]]
            # c = points[facets[0][2]]

            # cross = np.cross(a-b,c-b)

            i = 0
            cross = 0
            while np.sum(cross * cross) == 0:
                try:
                    a = points[facets[0][i+1]] - points[facets[0][i]]
                    # a = a/a.norm()
                    b = points[facets[0][i+2]] - points[facets[0][i]]
                    # b = b/b.norm()
                    cross = np.cross(a,b)
                    i += 1
                except:
                    return facets

            normal = np.array(region_normals[region[0]])
            for facet in this_region_facets:
                for vertex in facet:
                    point_normals[vertex] = normal

            if np.dot(cross,normal) < 0:
                t = []
                for i,face in enumerate(facets):
                    t.append(np.flip(face))
                return t
            else:
                return facets


        if not self.polygons_constructed:
            self.construct_polygons()

        self.logger.info('Save simplified surface mesh...')

        region_to_polygons = defaultdict(list)
        region_normals = dict()
        polygons = []
        polygon_to_region = []
        npolygons = 0
        for e0, e1 in self.graph.edges:

            c0 = self.graph.nodes[e0]
            c1 = self.graph.nodes[e1]

            if c0["occupancy"] != c1["occupancy"]:

                intersection_points = self._get_intersection(e0,e1)

                plane_id = self.graph[e0][e1]["supporting_plane_id"]
                plane = self.vg.input_planes[plane_id]
                correct_order = self._sort_vertex_indices_by_angle_exact(intersection_points,plane)

                assert(len(intersection_points)==len(correct_order))
                intersection_points = intersection_points[correct_order]

                ## orient polygon
                outside = self.cells.get(e0).center() if c1["occupancy"] else self.cells.get(e1).center()
                flip, normal = self._orient_polygon_exact(intersection_points,outside,return_orientation=True)
                if flip:
                    intersection_points = np.flip(intersection_points, axis=0)
                    normal = -normal
                region_normals[plane_id] = normal

                polygons.append(intersection_points.astype(np.float64))

                # region_to_polygons
                region_to_polygons[plane_id].append(npolygons)

                # polygon_to_region
                polygon_to_region.append(plane_id)

                npolygons+=1

        points = np.concatenate(polygons).astype(np.float64)
        points = np.unique(points, axis=0)

        ## index the facets of the mesh
        facets = []
        for poly in polygons:
            face = []
            for pt in poly:
                face.append(np.argwhere((np.equal(points, pt, dtype=object)).all(-1))[0][0])
            facets.append(face)
        assert(len(facets)==len(polygons))

        # ## debug export the unsimplified surface
        # if self.debug_export:
        #     self.complexExporter.write_surface_to_off(out_file[:-4]+"_debug.off", points=points, facets=facets)

        ## mark corner vertices
        vertex_is_corner = defaultdict(set)
        vertex_is_corner_array = np.zeros(points.shape[0],dtype=int) # used for reindexing
        for i,face in enumerate(facets):
            for vertex in face:
                vertex_is_corner[vertex].add(polygon_to_region[i])

        # init a surface extractor
        se=pdse(0)
        region_facets = []
        point_normals = np.ones(shape=points.shape)
        face_colors = []
        for region in region_to_polygons.items():
            this_region_facets = []
            boundary = _get_region_borders(region)
            # ss.get_cycles returns every cycle twice, once in each region.
            cycles = se.get_cycles(boundary)
            i = 0
            while i < len(cycles):
                if simplify_edges:
                    this_cycle = []
                    for c in cycles[i]:
                        if len(vertex_is_corner[c]) > 2:
                            this_cycle.append(c)
                            vertex_is_corner_array[c] = 1
                    this_region_facets.append(this_cycle)
                else:
                    this_region_facets.append(cycles[i])
                i+=2

            if triangulate: # triangulate all faces
                plane = PyPlane(self.vg.input_planes[region[0]])
                points2d = plane.to_2d(points)
                this_region_facets, _ = se.get_cdt_of_regions_with_holes(points2d, this_region_facets)
            elif len(this_region_facets) > 1: # triangulate only faces that have a whole
                    plane = PyPlane(self.vg.input_planes[region[0]])
                    points2d = plane.to_2d(points)
                    triangle_region_facets, region_has_hole = se.get_cdt_of_regions_with_holes(points2d, this_region_facets)
                    if region_has_hole:
                        this_region_facets = triangle_region_facets
                    else:
                        t = []
                        for cycle in this_region_facets:
                            t.append(cycle[:-1])
                        this_region_facets = t
            else:
                this_region_facets = [this_region_facets[0][:-1]]

            ## change orientation of the region if necessary
            this_region_facets = _orient_facets(this_region_facets, region)
            region_facets += this_region_facets
            face_colors.append(np.repeat(self.vg.plane_colors[region[0],np.newaxis],len(this_region_facets),axis=0))

        region_facets = np.array(region_facets)
        face_colors = np.concatenate(face_colors)

        if simplify_edges:
            ## remove unreferenced vertices and reindex
            points = points[vertex_is_corner_array>0]
            point_normals = point_normals[vertex_is_corner_array>0]
            vertex_is_corner_array[vertex_is_corner_array>0] = np.arange(vertex_is_corner_array.sum())
            t = []
            for facet in region_facets:
                t.append(vertex_is_corner_array[np.array(facet)])
            region_facets = t

        if backend == "python":
            self.logger.error(
                "Not all faces of the polygon mesh may be oriented correctly. Export a triangle mesh if you need correct orientation.")
            self.complexExporter.write_surface_to_off(out_file, points=points, facets=region_facets)
        elif backend == "trimesh":
            if not triangulate:
                self.logger.error("backend 'trimesh' only works with triangulate = True. Choose backend 'python' for exporting a polygon mesh.")
                raise NotImplementedError

            # TODO: pass the plane color through to the mesh. do this in all mesh export functions.
            mesh = trimesh.Trimesh(vertices=points, faces=region_facets, face_colors=face_colors)
            # trimesh.repair.fix_winding(mesh)
            # trimesh.repair.fix_normals(mesh)
            # trimesh.repair.fix_inversion(mesh)
            # mesh.fix_normals()
            mesh.export(out_file)
        else:
            self.logger.error(
                "{} is not a valid surface extraction backend. Choose either 'trimesh' or 'python'.".format(
                    backend))
            raise NotImplementedError

        # self.complexExporter.write_surface_to_ply(out_file, points=points, pnormals=point_normals, facets=region_facets)


    def save_surface(self, out_file, backend="python", triangulate=False, stitch_borders=True):

        """
        Extracts a watertight surface mesh from the labelled polyhedral complex.

        :param out_file: File to store the mesh.

        :param backend: Backend for surface extraction: 'python_exact','python', 'trimesh' or 'cgal'.
                        'python_exact': Extracts a polygon mesh. Works with an exact number type (SAGE rational). It guarantees to extract a watertight surface. The surface may have non-manifold edges. The extraction is slow due to the use of exact numbers.
                        'python': Extracts a polygon mesh. Similar to 'python_exact', but converts from exact to floating point numbers before surface assembly. This speeds up the extraction a lot but may lead to non-watertightness in rare cases.
                        'trimesh': Extracts a triangle mesh. Similar to 'python', but uses trimesh for assembling and storing the mesh.
                        'cgal': Can extract a polygon or triangle mesh. It is the fastest backend. Because the CGAL polygon mesh does not allow non-manifold edges, they are repair. This may lead to non-watertightness in some cases, especially if the mesh is not triangulated.

        :param triangulate: Flag that controls if mesh should be triangulated or not. Only taken into account for backend == 'cgal'.

        :param stitch_borders: Flag that controls if border edges should be stitch. Only taken into account for backend == 'cgal'.
        """


        if not self.polygons_constructed:
            self.construct_polygons()

        self.logger.info('Save surface mesh ({})...'.format(backend))


        polygons_exact = []
        points_exact = []
        polygons = []
        # for cgal export
        faces = []
        face_lens = []
        n_points = 0
        for e0, e1 in self.graph.edges:

            c0 = self.graph.nodes[e0]
            c1 = self.graph.nodes[e1]

            if c0["occupancy"] != c1["occupancy"]:

                intersection_points = self._get_intersection(e0,e1)

                plane_id = self.graph[e0][e1]["supporting_plane_id"]
                plane = self.vg.input_planes[plane_id]
                correct_order = self._sort_vertex_indices_by_angle_exact(intersection_points,plane)

                assert(len(intersection_points)==len(correct_order))
                intersection_points = intersection_points[correct_order]

                ## orient polygon
                outside = self.cells.get(e0).center() if c1["occupancy"] else self.cells.get(e1).center()
                if self._orient_polygon_exact(intersection_points,outside):
                    intersection_points = np.flip(intersection_points, axis=0)

                for i in range(intersection_points.shape[0]):
                    points_exact.append(tuple(intersection_points[i,:]))
                polygons_exact.append(intersection_points)

                polygons.append(intersection_points.astype(np.float64))

                # for cgal export
                faces.append(np.arange(len(intersection_points))+n_points)
                face_lens.append(len(intersection_points))
                n_points+=len(intersection_points)

        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        ## TODO: replace the CGAL backend with PDSE and add PDSE to COMPOD.
        if backend == "cgal":
            try:
                from pycompose import pdse
            except:
                self.logger.error("Could not import pdse. Either install COMPOSE from https://github.com/raphaelsulzer/compod#compose"
                                  " or use 'python' or 'trimesh' as backend.")
                raise ModuleNotFoundError
            se = pdse(verbosity=0,debug_export=False)
            se.load_soup(np.array(points_exact,dtype=np.float64), np.array(face_lens,dtype=int))
            se.soup_to_mesh(triangulate=triangulate,stitch_borders=True)
            se.save_mesh(out_file)

        elif backend == "python":
            if triangulate:
                self.logger.warning("Mesh will not be triangulated. Use backend 'cgal' or 'trimesh' for exporting a triangle mesh.")
            points = np.concatenate(polygons).astype(np.float64)
            points = np.unique(points, axis=0)

            facets = []
            for poly in polygons:
                face = []
                for pt in poly:
                    face.append(np.argwhere((np.equal(points, pt, dtype=object)).all(-1))[0][0])
                facets.append(face)
            self.complexExporter.write_surface_to_off(out_file, points=points, facets=facets)

        elif backend == "python_exact":
            if triangulate:
                self.logger.warning("Mesh will not be triangulated. Use backend 'cgal' or 'trimesh' for exporting a triangle mesh.")
            pset = set(points_exact)
            pset = np.array(list(pset),dtype=object)
            facets = []
            for poly in polygons_exact:
                face = []
                for pt in poly:
                    face.append(np.argwhere((np.equal(pset,pt,dtype=object)).all(-1))[0][0])
                facets.append(face)
            self.complexExporter.write_surface_to_off(out_file,points=np.array(pset,dtype=np.float32),facets=facets)

        elif backend == "trimesh":
            if not triangulate:
                self.logger.warning("Mesh will be triangulated. Use backend 'python' or 'cgal' for exporting a polygon mesh.")
            def _triangulate_points(v):
                n = len(v)
                triangles = []
                for i in range(n - 2):
                    tri = [0, i % n + 1, i % n + 2]
                    triangles.append(v[tri])
                return triangles
            # triangulate all faces
            tris = []
            for fa in faces:
                tris.append(_triangulate_points(fa))
            mesh = trimesh.Trimesh(vertices=points_exact,faces=np.concatenate(tris))
            mesh.export(out_file)
        else:
            self.logger.error("{} is not a valid surface extraction backend. Valid backends are 'python', 'python_exact', 'trimesh' and 'cgal'.".format(backend))
            raise NotImplementedError


    def save_in_cells_explode(self,out_file,shrink_percentage=0.01):

        self.logger.info('Save exploded inside cells...')

        os.makedirs(os.path.dirname(out_file),exist_ok=True)
        f = open(out_file,'w')

        def filter_node(node_id):
            return self.graph.nodes[node_id]["occupancy"]

        outverts = []
        facets = []
        vert_count = 0
        view = nx.subgraph_view(self.graph,filter_node=filter_node)

        # get total volume
        cell_volumes = []
        for node in view.nodes():
            vol = self.cells.get(node).volume()
            cell_volumes.append(vol)
        cell_volumes = np.array(cell_volumes)
        # cell_volumes = np.interp(cell_volumes, (cell_volumes.min(), cell_volumes.max()), (0.85, 0.80))
        cell_volumes = np.interp(cell_volumes, (cell_volumes.min(), cell_volumes.max()), (0.85, 0.95))
        bbox = np.array(self.bounding_poly.bounding_box(),dtype=float)
        bbox_dim = bbox[0, :] - bbox[1, :]
        # color_scales = (255-100)/bbox_dim
        # color_scales = [1.0,0.91,0.61]/bbox_dim
        bb_diag = np.linalg.norm(bbox_dim)
        try:
            from fancycolor.color import FancyColor
            col=FancyColor(bbox)
        except:
            col=None
        for i,node in enumerate(view.nodes()):
            # c = np.random.randint(low=100,high=255,size=3)
            polyhedron = self.cells.get(node)
            ss = polyhedron.render_solid().obj_repr(polyhedron.render_solid().default_render_params())
            verts = []
            for v in ss[2]:
                v = v.split(' ')
                verts.append(np.array(v[1:],dtype=float))

            verts = np.vstack(verts)
            centroid = np.mean(verts,axis=0)

            # vectors = (verts - centroid)*cell_volumes[i]
            vectors = (verts - centroid)
            vector_lens = np.linalg.norm(vectors,axis=1)
            max_vector = vector_lens.max()
            max_vector_ratio = (max_vector - bb_diag*shrink_percentage)/max_vector
            verts = centroid + vectors*max_vector_ratio
            # c = (np.abs(color_scales*centroid)+100).astype(int)
            # c = np.abs(color_scales*centroid)+[0.0,0.09,0.39]
            # c = np.array(colorsys.hsv_to_rgb(c[0],c[1],c[2]))*255
            # c = c.astype(int)
            c = col.get_rgb_from_xyz(centroid) if col is not None else np.random.randint(100,255,size=3)
            for v in verts:
                outverts.append([v[0], v[1], v[2], c[0], c[1], c[2]])

            for fa in ss[3]:
                tf = []
                for ffa in fa[2:].split(' '):
                    tf.append(str(int(ffa) + vert_count -1) + " ")
                facets.append(tf)
            vert_count+=len(ss[2])

        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment : in_cells:{}\n".format(len(view.nodes)))
        f.write("element vertex {}\n".format(len(outverts)))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("element face {}\n".format(len(facets)))
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in outverts:
            f.write("{} {} {} {} {} {}\n".format(v[0],v[1],v[2],v[3],v[4],v[5]))
        for fa in facets:
            f.write("{} ".format(len(fa)))
            for v in fa:
                f.write("{}".format(v))
            f.write("\n")

        f.close()


    def save_in_cells(self,out_file):

        self.logger.info('Save inside cells...')

        os.makedirs(os.path.dirname(out_file),exist_ok=True)
        f = open(out_file,'w')

        def filter_node(node_id):
            return self.graph.nodes[node_id]["occupancy"]

        verts = []
        facets = []
        vert_count = 0
        view = nx.subgraph_view(self.graph,filter_node=filter_node)
        # for node in enumerate(self.graph.nodes(data=True)):
            # if node[1]["occupancy"] == 1:
        for node in view.nodes():
            c = np.random.randint(low=100,high=255,size=3)
            polyhedron = self.cells.get(node)
            ss = polyhedron.render_solid().obj_repr(polyhedron.render_solid().default_render_params())
            for v in ss[2]:
                v = v.split(' ')
                verts.append([v[1], v[2], v[3], str(c[0]), str(c[1]), str(c[2])])

            for fa in ss[3]:
                tf = []
                for ffa in fa[2:].split(' '):
                    tf.append(str(int(ffa) + vert_count -1) + " ")
                facets.append(tf)
            vert_count+=len(ss[2])

        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment : in_cells:{}\n".format(len(view.nodes)))
        f.write("element vertex {}\n".format(len(verts)))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("element face {}\n".format(len(facets)))
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in verts:
            f.write("{} {} {} {} {} {}\n".format(v[0],v[1],v[2],v[3],v[4],v[5]))
        for fa in facets:
            f.write("{} ".format(len(fa)))
            for v in fa:
                f.write("{}".format(v))
            f.write("\n")

        f.close()


    def save_partition(self, filepath, rand_colors=True, export_boundary=True, with_primitive_id=True):
        """
        Save polygon soup of indexed convexes to a ply file.

        Parameters
        ----------
        filepath: str or Path
            Filepath to save obj file
        indices_cells: (n,) int
            Indices of cells to save to file
        use_mtl: bool
            Use mtl attribute in obj if set True
        """

        def _sorted_vertex_indices(adjacency_matrix):

            pointer = 0
            sorted_ = [pointer]
            for _ in range(len(adjacency_matrix[0]) - 1):
                connected = np.where(adjacency_matrix[pointer])[0]  # two elements
                if connected[0] not in sorted_:
                    pointer = connected[0]
                    sorted_.append(connected[0])
                else:
                    pointer = connected[1]
                    sorted_.append(connected[1])
            return sorted_


        self.logger.info('Save partition...')
        self.logger.debug("to {}".format(filepath))


        # create the dir if not exists
        if not self.polygons_initialized:
            self._init_polygons()


        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        points = []
        indices = []
        primitive_ids = []
        count = 0
        ecount = 0

        colors = []
        for c0,c1 in self.graph.edges:


            if not export_boundary:
                bbe = self.graph.edges[c0,c1].get("bounding_box_edge",False)
                if bbe:
                    continue


            ecount+=1
            face = self.graph.edges[c0,c1]["intersection"]
            verts = face.vertices_list()
            correct_vertex_order = _sorted_vertex_indices(face.adjacency_matrix())
            points.append(np.array(verts)[correct_vertex_order])
            # points.append(verts)
            indices.append(list(np.arange(count,len(verts)+count)))


            # plane = self.graph.edges[c0,c1]["supporting_plane"]
            plane_id = self.graph.edges[c0,c1]["supporting_plane_id"]
            ## if plane_id is negative append negative primitive_id


            if plane_id > -1:
                colors.append(self.vg.plane_colors[plane_id])
                # primitive_ids.append([self.vg.plane_order[plane_id]])
                pid = self.vg.plane_order[plane_id]
                primitive_ids.append(list(self.vg.polygons_from_plane[pid]))
            else:
                colors.append(np.random.randint(100, 255, size=3))
                primitive_ids.append([])


            count+=len(verts)

        if rand_colors:
            colors = np.random.randint(100, 255, size=(len(self.graph.edges), 3))


        points = np.concatenate(points)

        f = open(str(filepath),"w")
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment number_of_cells {}\n".format(len(self.graph.nodes)))
        # f.write("comment number_of_faces {}\n".format(len(self.graph.edges)))
        f.write("comment number_of_faces {}\n".format(ecount))
        f.write("element vertex {}\n".format(len(points)))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        # f.write("property uchar red\n")
        # f.write("property uchar green\n")
        # f.write("property uchar blue\n")
        # f.write("element face {}\n".format(len(self.graph.edges)))
        f.write("element face {}\n".format(ecount))
        f.write("property list uchar int vertex_indices\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        if with_primitive_id:
            f.write("property list uchar int primitive_indices\n")
        # f.write("property float a\n")
        # f.write("property float b\n")
        # f.write("property float c\n")
        # f.write("property float d\n")
        f.write("end_header\n")
        for v in points:
            f.write("{:.3f} {:.3f} {:.3f}\n".format(v[0],v[1],v[2]))
        for i,fa in enumerate(indices):
            f.write("{} ".format(len(fa)))
            for v in fa:
                f.write("{} ".format(v))
            c = colors[i]
            f.write("{} {} {}".format(c[0],c[1],c[2]))
            if with_primitive_id:
                pi = primitive_ids[i]
                f.write(" {} ".format(len(pi)))
                for v in pi:
                    f.write("{} ".format(v))
            # sp = supporting_planes[i]
            # f.write("{} {} {} {}".format(sp[0],sp[1],sp[2],sp[3]))
            f.write("\n")


        f.close()

    def _polygon_area(self,poly):
        """Compute polygon area, from here: https://stackoverflow.com/a/12643315"""

        def unit_normal(a, b, c):
            def det(a):
                return a[0][0] * a[1][1] * a[2][2] + a[0][1] * a[1][2] * a[2][0] + a[0][2] * a[1][0] * a[2][1] - a[0][
                    2] * \
                       a[1][1] * a[2][0] - a[0][1] * a[1][0] * a[2][2] - a[0][0] * a[1][2] * a[2][1]

            x = det([[1, a[1], a[2]],
                     [1, b[1], b[2]],
                     [1, c[1], c[2]]])
            y = det([[a[0], 1, a[2]],
                     [b[0], 1, b[2]],
                     [c[0], 1, c[2]]])
            z = det([[a[0], a[1], 1],
                     [b[0], b[1], 1],
                     [c[0], c[1], 1]])
            magnitude = (x ** 2 + y ** 2 + z ** 2) ** .5
            return (x / magnitude, y / magnitude, z / magnitude)


        if len(poly) < 3:  # not a plane - no area
            return 0

        total = [0, 0, 0]
        for i in range(len(poly)):
            vi1 = poly[i]
            if i is len(poly) - 1:
                vi2 = poly[0]
            else:
                vi2 = poly[i + 1]
            prod = np.cross(vi1, vi2)
            total[0] += prod[0]
            total[1] += prod[1]
            total[2] += prod[2]
        result = np.dot(total, unit_normal(poly[0], poly[1], poly[2]))
        return abs(result / 2)


    def _beta_skeleton(self, center, point, normal):

        co_tangent = center - point
        co_tangent = co_tangent / np.linalg.norm(co_tangent)

        normal = normal / np.linalg.norm(normal)

        return np.dot(normal,co_tangent)


    def _graph_cut(self,occs,binary_weight=0.2,binary_type="cc"):

        # unfortunately necessary for computing edge areas
        if binary_type == "area":
            self._init_polygons()

        graph = nx.convert_node_labels_to_integers(self.graph)

        labels = (occs[:,0]<occs[:,1]).astype(np.int32)

        self.logger.info("Apply occupancy regularization with graph cut (λ={:.2g})...".format(binary_weight))

        assert len(occs) == len(graph.nodes)

        dtype = np.float64
        # Internally, in the pyGCO code datatype is always converted to np.int32
        # I would need to use my own version of GCO (or modify the one used by pyGCO) to change that
        # Should probably be done at some point to avoid int32 overflow for larger scenes.


        gc = gco.GCO()
        gc.create_general_graph(len(graph.nodes), 2, energy_is_float=True)
        # data_cost = F.softmax(prediction, dim=-1)

        scale = 4

        # data_cost = prediction
        data_cost = np.array(occs, dtype=dtype)*scale

        ### append high cost for inside for infinite cell
        # data_cost = np.append(data_cost, np.array([[-10000, 10000]]), axis=0)
        gc.set_data_cost(data_cost)
        smooth = (1 - np.eye(2)).astype(dtype)
        gc.set_smooth_cost(smooth)

        edges = np.array(graph.edges)

        if binary_type == "cc":
            edge_weight = np.ones(edges.shape[0], dtype=dtype)*scale/edges.shape[0]
        elif binary_type == "area":
            edge_weight = []
            for e0, e1 in self.graph.edges:
                edge = self.graph.edges[e0, e1]
                vertices = np.array(edge["intersection"].vertices())
                assert (len(vertices))
                vertices = vertices[self._sorted_vertex_indices(edge["intersection"].adjacency_matrix())]
                edge_weight.append(self._polygon_area(vertices))
                # edge_weight.append(edge["intersection"].affine_hull_projection().volume().numerical_approx())
            edge_weight = np.array(edge_weight)/sum(edge_weight)
        elif binary_type == "beta_skeleton":
            edge_weight = []
            for e0, e1 in self.graph.edges:
                edge = self.graph.edges[e0,e1]
                normal = PyPlane(self.vg.input_planes[edge["supporting_plane_id"]]).normal
                point = np.array(edge["intersection"].vertices_list()[0])
                ew = 1 - min(self._beta_skeleton(self.cells[e0].center(),point,normal),
                             self._beta_skeleton(self.cells[e1].center(),point,-normal))
                edge_weight.append(ew)
            edge_weight = np.array(edge_weight)/sum(edge_weight)
        else:
            self.logger.error("{} is not a valid binary type".format(binary_type))
            raise NotImplementedError
        
        # TODO: try to solve orientation with the graph cut
        # also add

        # e1 = np.append(edges[:,1],edges[:,0],axis=0)
        # e2 = np.append(edges[:,0],edges[:,1],axis=0)
        # ew = np.append(edge_weight,edge_weight,axis=0)
        # gc.set_all_neighbors(e1, e2, edge_weight * binary_weight * scale)

        gc.set_all_neighbors(edges[:, 0], edges[:, 1], edge_weight * binary_weight * scale)

        for i, l in enumerate(labels):
            gc.init_label_at_site(i, l)

        self.logger.debug("Energy before GC (D + λ*S): {:.2g} + {:.2g}*{:.2g} = {:.2g}".format(gc.compute_data_energy(), binary_weight, gc.compute_smooth_energy()/binary_weight,
                                                            gc.compute_data_energy()+gc.compute_smooth_energy()))

        gc.expansion()
        self.logger.debug("Energy after GC (D + λ*S): {:.2g} + {:.2g}*{:.2g} = {:.2g}".format(gc.compute_data_energy(), binary_weight, gc.compute_smooth_energy()/binary_weight,
                                                            gc.compute_data_energy() + gc.compute_smooth_energy()))

        labels = gc.get_labels()

        return labels


    def label_partition(self,mode="normals",mesh_file=None,n_test_points=50,binary_weight=0.0,binary_type="cc",footprint=None,z_range=None):

        self.logger.info('Label {} cells with {}...'.format(len(self.graph.nodes),mode))

        if mode == "normals":
            if self.vg.points_type == "samples":
                self.logger.error("Cannot label partition with normals from sampled points. Please choose mode 'mesh' instead.")
                raise NotImplementedError
            occs = self.label_partition_with_point_normals(footprint=footprint,z_range=z_range)
        elif mode == "mesh":
            if mesh_file is None:
                self.logger.error("Please provide a closed mesh_file to label partition with a mesh.")
                raise ValueError
            occs = self.label_partition_with_mesh(mesh_file,n_test_points)
        else:
            self.logger.error("{} is not a valid labelling type. Choose either 'pc', 'mesh' or 'load'.".format(mode))
            raise NotImplementedError

        if binary_weight > 0.0:
            # occs = self.graph_cut(occs)
            # compute regularization weights
            occs = self._graph_cut(occs,binary_weight=binary_weight,binary_type=binary_type)
        else:
            occs = occs[:,0]>occs[:,1]

        assert len(occs) == len(self.graph.nodes), "Number of cccupancy labels and graph nodes is not the same."
        occs = dict(zip(self.graph.nodes, np.rint(occs).astype(int)))

        nx.set_node_attributes(self.graph,occs,"occupancy")

        self.partition_labelled = True

        return 0


    def label_partition_with_mesh(self, mesh_file, n_test_points=50):
        """
        Compute occupancy of each cell in the partition by sampling points inside each cells and checking if they lie inside the provided reference mesh.

        :param mesh_file: The input mesh file.
        :param n_test_points: Number of test points per cell.
        """

        try:
            from pycompose import pdl
        except:
            self.logger.error("Labeling partition with 'mesh' not available. Either install COMPOSE from https://github.com/raphaelsulzer/compod#compose"
                              " or use type 'normals' for labeling.")
            raise ModuleNotFoundError

        points = []
        points_len = []
        # for i,id in enumerate(list(self.graph.nodes)):
        for id,cell in self.cells.items():
            # if id < 0:  continue # skip the bounding box cells
            bb = self.graph.nodes[id].get("bounding_box",0)
            if bb:  continue # skip the bounding box cells
            # cell = self.cells.get(id)
            if self.debug_export:
                out_path = os.path.join(self.debug_export,"final_cells")
                self.complexExporter.write_cell(out_path,cell,count=id)
            pts = np.array(cell.vertices())
            points.append(pts)
            # print(pts)
            points_len.append(pts.shape[0])


        # assert(isinstance(points[0].dtype,np.float32))
        pl=pdl(n_test_points)
        if pl.load_mesh(mesh_file):
            return 1
        occs = pl.label_cells(np.array(points_len),np.concatenate(points,axis=0))
        del pl

        occs = np.hstack((occs,np.zeros(6))) # this is for the bounding box cells, which are the last 6 cells of the graph
        occs = np.array([occs,1-occs]).transpose()

        return occs


    def label_partition_with_point_normals(self,footprint=None,z_range=None):
        """
        Compute the occupancy of each cell of the partition according to the normal criterion introduced in Kinetic Shape Reconstruction [Bauchet & Lafarge 2020] (Section 4.2).
        """

        def collect_facet_points():

            point_ids_dict = dict()
            count = 0
            # edge_areas_dict = dict()
            for e0, e1 in self.graph.edges:

                edge = self.graph.edges[e0, e1]
                if edge["supporting_plane_id"] < 0:  # bounding box planes have no associated points, so skip them
                    point_ids_dict[(e0, e1)] = np.empty(shape=0, dtype=np.int32)
                    continue

                group = self.vg.groups[edge["group_id"]]

                ## use shapely contains_xy with projected points.
                vertices = np.array(edge["intersection"].vertices())
                assert (len(vertices))
                vertices = vertices[self._sorted_vertex_indices(edge["intersection"].adjacency_matrix())]
                plane = PyPlane(self.vg.split_planes[edge["group_id"]])
                poly = Polygon(np.array(plane.to_2d(vertices)))
                contain = contains_xy(poly, self.vg.projected_points[group])

                point_ids_dict[(e0, e1)] = group[contain]
                if self.debug_export:
                    pts = self.vg.points[group[contain]]
                    count += 1
                    col = np.random.randint(0, 255, size=3)
                    self.complexExporter.write_points(os.path.join(self.debug_export, "labelling_facets_group_points"),
                                                      self.vg.points[group],
                                                      count=str(count) + "c", color=col)
                    self.complexExporter.write_facet(os.path.join(self.debug_export, "labelling_facets"),
                                                     edge["intersection"],
                                                     count=str(count), color=col)
                    if len(pts):
                        self.complexExporter.write_facet(os.path.join(self.debug_export, "labelling_facets_containing"),
                                                         edge["intersection"],
                                                         count=str(count), color=col)
                        self.complexExporter.write_points(
                            os.path.join(self.debug_export, "labelling_facets_contained_points"), pts,
                            count=str(count) + "c", color=col)

            nx.set_edge_attributes(self.graph, point_ids_dict, "point_ids")


        def make_point_class_weight():

            # self.point_class_weights = np.ones(self.vg.classes.max() + 1)
            self.point_class_weights = np.ones(67+1)
            # {1: "Unclassified", 2: "Ground", 3: "Low_Vegetation", 4: "Medium_Vegetation", 5: "High_Vegetation",
            #  6: "Building", 9: "Water",
            #  17: "17", 63: "Etage", 64: "64", 65: "65", 66: "Floor", 67: "Walls"}
            self.point_class_weights[0] = 0
            self.point_class_weights[1] = 1
            self.point_class_weights[6] = 3
            self.point_class_weights[63] = 0
            self.point_class_weights[66] = 12
            self.point_class_weights[67] = 2

        def collect_node_votes(footprint=None, z_range=None):

            if len(self.vg.classes):
                make_point_class_weight()

            occs = []
            if footprint is not None:
                footprint = Polygon(footprint)

            all_cell_points = []
            all_cell_normals = []
            type_colors = []
            for node in self.graph.nodes:

                centroid = np.array(self.cells[node].vertices()).mean(axis=0)
                footprint_weight = 24
                n = footprint_weight * 2 * len(self.vg.points) * self.point_class_weights.max() / len(self.cells)

                if self.graph.nodes[node].get("bounding_box", 0):
                    # occupancy_dict[node] = (0,1000)
                    occs.append([0, n])  # TODO: this should probably be related to the number of points or something
                    type_colors.append([205, 82, 61])
                elif footprint is not None and not contains_xy(footprint, x=centroid[0], y=centroid[1]):
                    occs.append([0, n])
                    type_colors.append([147, 196, 125])
                elif z_range is not None and z_range[0] <= centroid[2] and z_range[1] >= centroid[2] and contains_xy(
                        footprint, x=centroid[0], y=centroid[1]):
                    occs.append([n, 0])
                    type_colors.append([61, 184, 205])
                elif z_range is not None and centroid[2] > z_range[2]:
                    occs.append([0, n])
                    type_colors.append([205, 61, 112])
                else:
                    type_colors.append([241, 194, 50])
                    inside_weight = 0;
                    outside_weight = 0
                    cell_points = []
                    cell_normals = []
                    for edge in self.graph.edges(node):
                        edge = self.graph.edges[edge]
                        pts = self.vg.points[edge["point_ids"]]
                        if not len(pts):
                            continue
                        inside_vectors = centroid - pts
                        normal_vectors = self.vg.normals[edge["point_ids"]]
                        dp = (normal_vectors * inside_vectors).sum(axis=1)
                        iweight = (dp < 0).astype(float)
                        oweight = (dp > 0).astype(float)
                        if len(self.vg.classes):
                            class_weights = self.vg.classes[edge["point_ids"]]
                            class_weights = self.point_class_weights[class_weights].astype(float)
                            iweight *= class_weights
                            oweight *= class_weights
                        inside_weight += iweight.sum()
                        outside_weight += oweight.sum()
                        if self.debug_export:
                            cell_points.append(pts)
                            cell_normals.append(normal_vectors)

                    # occupancy_dict[node] = (inside_weight,outside_weight)
                    occs.append([inside_weight, outside_weight])

            if len(self.vg.classes):
                occs = np.array(occs) / (2 * self.point_class_weights[self.vg.classes].sum())
            else:
                occs = np.array(occs) / (2 * len(self.vg.points))

            if self.debug_export:

                from fancycolor import GradientColor2D

                inside_weight = occs[:, 0]
                outside_weight = occs[:, 1]

                icol = GradientColor2D("Reds", inside_weight.min(), inside_weight.max()).get_rgb(inside_weight)
                ocol = GradientColor2D("Greens", outside_weight.min(), outside_weight.max()).get_rgb(outside_weight)

                for i, node in enumerate(self.graph.nodes):
                    st = "out" if inside_weight[i] <= outside_weight[i] else "in"
                    color = np.array(ocol[i])[:3] if inside_weight[i] <= outside_weight[i] else np.array(icol[i][:3])

                    self.complexExporter.write_cell(os.path.join(self.debug_export, "labelling_cells_{}".format(st)),
                                                    self.cells[node],
                                                    count=str(node) + st, color=color)
                    color = type_colors[i]
                    self.complexExporter.write_cell(
                        os.path.join(self.debug_export, "labelling_cells_type_{}".format(st)), self.cells[node],
                        count=str(node) + st, color=color)
                    # self.complexExporter.write_points(os.path.join(self.debug_export,"labelling_points"),color=color, count=str(node)+st,
                    #                                       points=np.concatenate(cell_points),normals=np.concatenate(cell_normals))

            return occs

        collect_facet_points()
        return collect_node_votes(footprint=footprint,z_range=z_range)



    def _get_best_split(self,current_ids,primitive_dict,insertion_order="product-earlystop"):
        """
        CPU version of _get_best_split_gpu().
        Note: This function could also be written with a single loop and np.tensordot, just like _get_best_split_gpu, but it will make it actually slightly slower.

        :param current_ids:
        :param primitive_dict:
        :param insertion_order:
        :return:
        """

        earlystop = False
        if "earlystop" in insertion_order:
            earlystop = True

        planes = self.vg.split_planes[current_ids]
        hull_verts = self.vg.hull_vertices[current_ids]

        left_right = []
        for i,id in enumerate(current_ids):
            left = 0; right = 0; intersect = 0
            pl = planes[i]
            for j,id2 in enumerate(current_ids):
                if i == j: continue
                pts = self.vg.points[hull_verts[j]]
                which_side = np.dot(pl[:3],pts.transpose())
                which_side = (which_side < -pl[3])
                tleft = (which_side).all(axis=-1)
                if tleft:
                    left+=1
                    continue
                tright = (~which_side).all(axis=-1)
                if tright:
                    right+=1
                else:
                    intersect+=1

            # assert(left + right + intersect == current_ids.shape[0]-1)
            if earlystop:
                # if left == current_ids.shape[0]-1 or right == current_ids.shape[0]-1:
                if intersect == 0:
                    return i

            left_right.append([left, right, intersect])

        left_right = np.array(left_right)
        if "sum" in insertion_order:
            left_right = np.sum(left_right[:,:2],axis=1)
            best_plane_id = np.argmax(left_right)
        elif "product" in insertion_order:
            left_right = np.prod(left_right[:,:2],axis=1)
            best_plane_id = np.argmax(left_right)
        elif "intersect" in insertion_order:
            left_right = np.abs(left_right[:,0]-left_right[:,1])+left_right[:,2]
            best_plane_id = np.argmin(left_right)
        elif "equal" in insertion_order:
            left_right = np.abs(left_right[:,0]-left_right[:,1])
            best_plane_id = np.argmin(left_right)
        else:
            raise NotImplementedError

        return best_plane_id


    def _get_best_split_gpu(self,current_ids,primitive_dict,insertion_order="product-earlystop"):


        earlystop = False
        if "earlystop" in insertion_order:
            earlystop = True

        planes = self.vg.split_planes[current_ids]
        hull_verts = self.vg.hull_vertices[current_ids]
        points = self.vg.points[hull_verts]
        points =  self.torch.from_numpy(points).type(self.torch.float32).to('cuda').transpose(2,0)
        planes = self.torch.from_numpy(planes).type(self.torch.float32).to('cuda')

        ### find the plane which seperates all other planes without splitting them
        left_right = []
        for i,id in enumerate(current_ids):
            pl = planes[i]

            pts = self.torch.cat((points[:,:,:i],points[:,:,i+1:]),axis=2)    # take out the support points of plane[i]

            which_side = self.torch.tensordot(pl[:3], pts, 1)
            which_side = which_side < -pl[3]
            left = which_side.all(axis=0)
            right = (~which_side).all(axis=0)

            lr = self.torch.logical_or(left,right)

            if earlystop and lr.all():
                # self.logger.debug("earlystop")
                return i
            else:
                left = left.sum().item()
                right = right.sum().item()
                intersect = which_side.shape[1] - left - right

            left_right.append([left, right, intersect])

        left_right = np.array(left_right)
        if "sum" in insertion_order:
            left_right = np.sum(left_right[:,:2],axis=1)
            best_plane_id = np.argmax(left_right)
        elif "product" in insertion_order:
            left_right = np.prod(left_right[:,:2],axis=1)
            best_plane_id = np.argmax(left_right)
        elif "intersect" in insertion_order:
            left_right = np.abs(left_right[:,0]-left_right[:,1])+left_right[:,2]
            best_plane_id = np.argmin(left_right)
        elif "equal" in insertion_order:
            left_right = np.abs(left_right[:,0]-left_right[:,1])
            best_plane_id = np.argmin(left_right)
        else:
            raise NotImplementedError

        return best_plane_id


    def _get_best_plane(self,current_ids,insertion_order="product-earlystop"):
        """
        Get the best plane from the planes in the current cell (ie from current_ids).
        :param current_ids: The planes in the current cell.
        :param insertion_order: The insertion order type.
        :return: The ID of the best plane in the current cell according to insertion order type. The ID is relativ to the current_ids, ie for getting the global id do current_ids[ID].
        """

        pgs = None # this probably has to be self.vg.groups[current_ids] but never tested these functions

        if insertion_order == "random":
            return np.random.choice(len(current_ids),size=1)[0]
        elif insertion_order in ["product", "product-earlystop", "sum", "sum-earlystop", "equal", "equal-earlystop", "intersect", "intersect-earlystop"]:
            if self.device == "gpu":
                return self._get_best_split_gpu(current_ids,insertion_order)
            elif self.device == "cpu":
                return self._get_best_split(current_ids,insertion_order)
            else:
                raise NotImplementedError
        elif insertion_order == "n-points":
            npoints = []
            for pg in pgs:
                npoints.append(pg.shape[0])
            npoints = np.array(npoints)
            return np.argmax(npoints)
        elif insertion_order == 'volume':
            volumes = []
            for pg in pgs:
                volumes.append(np.prod(np.max(pg,axis=0)-np.min(pg,axis=0)))
            return np.argmax(volumes)
        elif insertion_order == 'norm':
            norms = []
            for pg in pgs:
                norms.append(np.linalg.norm(np.max(pg, axis=0) - np.min(pg, axis=0), ord=2, axis=0))
            return np.argmax(norms)
        elif insertion_order == 'area':
            areas = []
            for i,plane in enumerate(self.vg.split_planes[current_ids]):
                if pgs[i].shape[0] > 2:
                    mesh = PyPlane(plane).get_trimesh_of_projected_points(pgs[i])
                    areas.append(mesh.area)
                else:
                    areas.append(0)
            return np.argmax(areas)
        else:
            raise NotImplementedError


    def _split_support_points(self,best_plane_id,current_ids):

        '''
        Split all primitive (ie 2D convex hulls) in the current cell with the best plane.

        :param best_plane_id: The best plane in the current cell, according to self._get_best_plane().
        :param current_ids: All planes in the current cell.
        :param th: Threshold for minimum number of points of a primitive to be inserted.
        :return: IDs of primitives falling left and right of the best plane.
        '''

        assert self.insertion_threshold >= 0,"Threshold must be >= 0"

        best_plane = self.vg.split_planes[current_ids[best_plane_id]]


        ### now put the planes into the left and right subspace of the best_plane split
        ### planes that lie in both subspaces are split (ie their point_groups are split) and appended as new planes to the planes array, and added to both subspaces
        left_plane_ids = []
        right_plane_ids = []
        for id in current_ids:

            if id == current_ids[best_plane_id]:
                continue

            this_group = self.vg.groups[id]
            which_side = np.dot(best_plane[:3],self.vg.points[this_group].transpose())
            left_point_ids = this_group[which_side < -best_plane[3]]
            right_point_ids = this_group[which_side > -best_plane[3]]

            if (this_group.shape[0] - left_point_ids.shape[0]) <= self.insertion_threshold:
                left_plane_ids.append(id)
            elif(this_group.shape[0] - right_point_ids.shape[0]) <= self.insertion_threshold:
                right_plane_ids.append(id)
            else:

                # TODO: to make the partition exact (within floating point precision), I need to add the intersection of 'best plane' with the current convex hull, ie 'ProjectedConvexHull(self.vg.split_planes[id],self.vg.points[left_point_ids])'
                # one way to do this would be to:
                    # 1. compute the line of 'best_plane' intersect 'self.vg.split_planes[id]'
                    # 2. project the line to self.vg.split_planes[id]
                    # 3. intersect the line with the convex hull: https://stackoverflow.com/questions/30486312/intersection-of-nd-line-with-convex-hull-in-python

                if (left_point_ids.shape[0] > self.insertion_threshold):
                    left_plane_ids.append(self.vg.split_planes.shape[0])
                    self.vg.split_planes = np.vstack((self.vg.split_planes, self.vg.split_planes[id]))
                    self.vg.plane_ids.append(self.vg.plane_ids[id])
                    self.vg.halfspaces.append(self.vg.halfspaces[id])
                    self.vg.groups.append(left_point_ids)


                    # if not enough points for making a convex hull we simply keep the points 
                    
                    # get all hull points and  make a new hull on the left side
                    if left_point_ids.shape[0] > 2:
                        try:
                            new_hull = ProjectedConvexHull(self.vg.split_planes[id],self.vg.points[left_point_ids])
                            new_group = left_point_ids[new_hull.hull.vertices]
                        except:
                            # putting this except here, because even if there are more than 2 points, but they lie on the same line, you cannot get a convex hull.
                            new_group = left_point_ids
                    else:
                        new_group = left_point_ids

                    # fill the hull_vertices array to make it a matrix instead of jagged array for an efficient _get_best_plane function with matrix multiplications
                    # if torch.nested.nested_tensor ever supports broadcasting and dot products, the code could be simplified a lot.
                    if new_group.shape[0] <= self.vg.n_fill:
                        fill = self.vg.n_fill - new_group.shape[0]
                        fill = np.random.choice(left_point_ids,fill)
                        new_group = np.concatenate((new_group,fill))
                    else:
                        self.logger.warning(
                            "Fill value overflow. Len of hull points = {}, fill value = {}. Increase vg.n_fill in vg.py for more robustness.".format(new_group.shape[0],self.vg.n_fill))
                        new_group = new_group[:self.vg.n_fill]

                    self.vg.hull_vertices = np.vstack((self.vg.hull_vertices,new_group))

                if (right_point_ids.shape[0] > self.insertion_threshold):
                    right_plane_ids.append(self.vg.split_planes.shape[0])
                    self.vg.split_planes = np.vstack((self.vg.split_planes, self.vg.split_planes[id]))
                    self.vg.plane_ids.append(self.vg.plane_ids[id])
                    self.vg.halfspaces.append(self.vg.halfspaces[id])
                    self.vg.groups.append(right_point_ids)

                    # if not enough points for making a convex hull we simply keep the points

                    # get all hull points and  make a new hull on the right side
                    if right_point_ids.shape[0] > 2:
                        try:
                            new_hull = ProjectedConvexHull(self.vg.split_planes[id], self.vg.points[right_point_ids])
                            new_group = right_point_ids[new_hull.hull.vertices]
                        except:
                            # putting this except here, because even if there are more than 2 points, but they lie on the same line, you cannot get a convex hull.
                            new_group = right_point_ids
                    else:
                        new_group = right_point_ids

                    # fill the hull_vertices array to make it a matrix instead of jagged array for an efficient _get_best_plane function with matrix multiplications
                    # if torch.nested.nested_tensor ever supports broadcasting and dot products, the code could be simplified a lot.
                    if new_group.shape[0] <= self.vg.n_fill:
                        fill = self.vg.n_fill - new_group.shape[0]
                        fill = np.random.choice(right_point_ids, fill)
                        new_group = np.concatenate((new_group, fill))
                    else:
                        self.logger.warning(
                            "Fill value overflow. Len of hull points = {}, fill value = {}. Increase vg.n_fill in vg.py for more robustness.".format(new_group.shape[0],self.vg.n_fill))
                        new_group = new_group[:self.vg.n_fill]

                    self.vg.hull_vertices = np.vstack((self.vg.hull_vertices, new_group))

                self.split_count+=1

                # planes[id, :] = np.nan
                # point_groups[id][:, :] = np.nan

        return left_plane_ids,right_plane_ids


    def simplify_partition_graph_based(self):

        if not self.partition_labelled:
            self.logger.error("Partition has to be labelled with an occupancy per cell to be simplified.")
            return 0

        if not self.tree_simplified:
            self.logger.warning("This function is slow. It is recommended to call simplify_tree_based first.")

        # TODO: incooperate the function above into this one. everytime I deal with siblings, I do not have to compute the volume
            # simply need to update the tree with the correct node_id, how it is already done in simplify_partition_tree_based
            # furthermore, the case that I previously drew and found to be unsolveable with tree traversal is also solveable with tree traversal
                # if the two siblings of two graph adjacent nodes where split with the same plane ID they should be collapseable!?
        # TOOD: save the labelling to file. it takes the most amount of time when prototyping tree collapse and alos graph-cut later

        self.logger.info('Simplify partition (graph-based) with iterative neighbor collapse...')

        before = len(self.graph.nodes)
        nx.set_node_attributes(self.graph,None,"volume")
        nx.set_edge_attributes(self.graph,None,"union_volume")

        def filter_edge(c0, c1):
            return not self.graph.edges[c0, c1]["processed"]

        nx.set_edge_attributes(self.graph,False,"processed")
        edges = list(nx.subgraph_view(self.graph, filter_edge=filter_edge).edges)
        while len(edges):

            for c0, c1 in edges:

                if not (self.graph.nodes[c0]["occupancy"] == self.graph.nodes[c1]["occupancy"]):
                    self.graph.edges[c0,c1]["processed"] = True
                    continue

                cx = None
                if self.graph.edges[c0,c1]["union_volume"] is None:
                    cx = Polyhedron(vertices=self.cells[c0].vertices_list() + self.cells[c1].vertices_list())
                    self.graph.edges[c0,c1]["union_volume"] = cx.volume()
                if self.graph.nodes[c0]["volume"] is None: self.graph.nodes[c0]["volume"] = self.cells[c0].volume()
                if self.graph.nodes[c1]["volume"] is None: self.graph.nodes[c1]["volume"] = self.cells[c1].volume()
                if self.graph.edges[c0, c1]["union_volume"] != (self.graph.nodes[c0]["volume"]+self.graph.nodes[c1]["volume"]):
                    self.graph.edges[c0,c1]["processed"] = True
                    continue
                else:
                    self.graph.nodes[c0]["volume"] = self.graph.edges[c0, c1]["union_volume"]
                    nx.contracted_edge(self.graph, (c0, c1), self_loops=False, copy=False)
                    self.cells[c0] = cx if cx is not None else Polyhedron(
                        vertices=self.cells[c0].vertices_list() + self.cells[c1].vertices_list())
                    del self.cells[c1]
                    for n0, n1 in self.graph.edges(c0):
                        if (self.graph.nodes[n0]["occupancy"] == self.graph.nodes[n1]["occupancy"]):
                            self.graph.edges[n0, n1]["union_volume"] = \
                                Polyhedron(
                                    vertices=self.cells[n0].vertices_list() + self.cells[n1].vertices_list()).volume()
                            self.graph.edges[n0, n1]["processed"] = False
                    del self.graph.nodes[c0]["contraction"]

                    break


            edges = list(nx.subgraph_view(self.graph, filter_edge=filter_edge).edges)


        self.logger.info("Simplified partition from {} to {} cells".format(before, len(self.graph.nodes)))

        self.polygons_initialized = False


    def simplify_partition_tree_based(self):

        if not self.partition_labelled:
            self.logger.error("Partition has to be labelled with an occupancy per cell to be simplified.")
            return 0

        ### this is nice and very fast, but it cannot simplify every case. because the tree would need to be restructured.
        ### there are cases where two cells are on the same side of the surface, there union is convex, but they are not siblings in the tree -> they cannot be simplified with this function

        self.logger.info('Simplify partition (tree-based) with iterative sibling collapse...')

        def filter_edge(n0,n1):
            if self.graph.edges[n0,n1].get("bounding_box",0):
                return False
            # if n0 < 0 or n1 < 0:
            #     return False
            if not self.tree[n0].is_leaf():
                return False
            if not self.tree[n1].is_leaf():
                return False
            if len(self.tree.siblings(n0)) == 0:
                return False
            if self.tree.siblings(n0)[0].identifier != n1:
                return False
            to_process = (self.graph.nodes[n0]["occupancy"] == self.graph.nodes[n1]["occupancy"])
                          # and self.graph.edges[n0,n1]["convex_intersection"])
            return to_process

        before=len(self.graph.nodes)
        edges = list(nx.subgraph_view(self.graph,filter_edge=filter_edge).edges)
        while len(edges):

            for c0,c1 in edges:
                if not c1 in self.cells:
                    continue

                nx.contracted_edge(self.graph, (c0, c1), self_loops=False, copy=False)

                parent = self.tree.parent(c0)
                parent_parent = self.tree.parent(parent.identifier)
                if parent_parent is None:
                    continue

                self.tree.remove_node(parent.identifier)

                dd = {"plane_ids": parent.data["plane_ids"]}
                self.cells[c0] = Polyhedron(vertices=self.cells[c0].vertices_list()+self.cells[c1].vertices_list())
                del self.cells[c1]
                try:
                    self.tree.create_node(tag=c0, identifier=c0, data=dd, parent=parent_parent.identifier)
                except:
                    a=4
            edges = list(nx.subgraph_view(self.graph, filter_edge=filter_edge).edges)


        self.logger.info("Simplified partition from {} to {} cells".format(before,len(self.graph.nodes)))
        self.polygons_initialized = False
        self.tree_simplified = True


    def _init_polygons(self):

        """
        3. initialize the polygons
        3a. intersects all pairs of polyhedra that share an edge in the graph and store the intersections on the edge
        3b. init an empty vertices list needed for self.construct_polygons
        """

        self.logger.info("Initialise polygons...")

        for e0,e1 in self.graph.edges:

            edge = self.graph.edges[e0,e1]
            c0 = self.cells.get(e0)
            c1 = self.cells.get(e1)
            ### this doesn't work, because after simplify some intersections are set, but are set with the wrong intersection from a collapse.
            ### I could just say if self.simplified or something like that, but for now will just recompute all the intersections here
            # if not self.graph.edges[e0,e1]["intersection"]:
            intersection = c0.intersection(c1)
            if intersection.dim() == 2:
                edge["intersection"] = intersection
                edge["vertices"] =  intersection.vertices_list()
            else:
                self.graph.remove_edge(e0,e1)

            # edge["vertices"] = edge["intersection"].vertices_list()

        self.polygons_initialized = True

    def construct_polygons(self):

        """
        add missing vertices to the polyhedron facets by intersecting all neighbors with all neighbors
        :return:
        """

        if not self.polygons_initialized:
            self._init_polygons()

        self.logger.info("Construct polygons...")


        for c0,c1 in list(self.graph.edges):

            if self.graph.nodes[c0]["occupancy"] == self.graph.nodes[c1]["occupancy"]:
                continue

            current_edge = self.graph[c0][c1]
            current_facet = current_edge["intersection"]

            for neighbor in list(self.graph[c0]):
                if neighbor == c1: continue
                this_edge = self.graph[c0][neighbor]
                facet_intersection = current_facet.intersection(this_edge["intersection"])
                if facet_intersection.dim() == 0 or facet_intersection.dim() == 1:
                    current_edge["vertices"]+=facet_intersection.vertices_list()

            for neighbor in list(self.graph[c1]):
                if neighbor == c0: continue
                this_edge = self.graph[c1][neighbor]
                facet_intersection = current_facet.intersection(this_edge["intersection"])
                if facet_intersection.dim() == 0 or facet_intersection.dim() == 1:
                    current_edge["vertices"] += facet_intersection.vertices_list()

        self.polygons_constructed = True


    def partition_from_tree(self, model):

        def which_side(node_id):
            parent = self.tree.parent(node_id)
            return self.tree.children(parent.identifier)[0].identifier != node_id

        # TODO: make a version where the tree is build without any intersection computations.
        #  Here the tree nodes are planes, and only the leaf nodes are (not yet) constructed cells (see Murali et al. Figure 3 for such a tree).
        # To construct the cells, (after the tree has been constructed) one simply has to walk up the tree from the leaf node to the root and collect the planes along it.
        # Planes are oriented depending on which side of the tree one is coming from e.g. each child could have a label positive or negative.
        # Stop walking once the collected planes form a bounded region -> the cell.
        # Now the cells can be labelled and simplified (siblings are known from the tree) and a convex decomposition can be extracted.
        # For surface reconstruction, i.e. to get the interface facets, a graph adjacency has to be recovered.
        # This could maybe be done by analysing from which supporting planes the cells come, i.e. if two cells share one supporting plane they are probably heighbors.

        construction_planes = defaultdict(list)

        for cell in self.tree.leaves():

            id = cell.identifier
            id = self.tree.parent(id).identifier
            if self.tree[id].is_root():
                construction_planes[cell.identifier].append(self.tree[id].data["supporting_plane_id"])
                continue
            side = which_side(id)
            construction_planes[cell.identifier].append(self.tree[id].data["supporting_plane_id"])
            while (side == which_side(id)):
                id = self.tree.parent(id).identifier
                construction_planes[cell.identifier].append(self.tree[id].data["supporting_plane_id"])
                if self.tree[id].is_root():
                    break

        hspaces=[]
        for cell_id,planes in construction_planes.items():
            for plane in planes:
                hspaces.append(self._inequalities(self.planes[plane_id]))
            p=Polyhedron(ieqs=hspaces)


    def _insert_new_plane(self,parent,id,left_planes=[],right_planes=[]):

        current_cell = self.cells.get(parent)
        best_plane_id_input = self.vg.plane_ids[id]

        ## create the new convexes
        # hspace_positive, hspace_negative = self.vg.halfspaces[current_ids[best_plane_id],0], self.vg.halfspaces[current_ids[best_plane_id],1]
        hspace_positive, hspace_negative = self.vg.halfspaces[id][0], \
                                           self.vg.halfspaces[id][1]

        cell_negative = current_cell.intersection(hspace_negative)
        cell_positive = current_cell.intersection(hspace_positive)

        # # if not construct_partition and (cell_positive.is_empty() or cell_negative.is_empty()):
        # if (cell_positive.is_empty() or cell_negative.is_empty()):
        #     # when constructing the partition I expand a tree. so even if this is a meaningless split, i.e. one where one of the cells is empty,
        #     # I still need to add a new tree node, so the parent node gets visited again and another split plane is used (in case the cell contains more planes).
        #     # However, when inserting additional planes, I don't expand a tree, but loop over all the planes and cells, i.e. I can just pass the meaningless splits.
        #     return 0

        ## update tree by creating the new nodes with the planes that fall into it
        ## and update graph with new nodes
        if (cell_negative.dim() == 3):
            if self.debug_export:
                self.complexExporter.write_cell(os.path.join(self.debug_export, "construct_cells"), cell_negative,
                                                count=str(self.cell_count + 1) + "n")
            dd = {"plane_ids": np.array(left_planes)}
            self.cell_count += 1
            neg_cell_id = self.cell_count
            self.tree.create_node(tag=neg_cell_id, identifier=neg_cell_id, data=dd, parent=parent)
            self.graph.add_node(neg_cell_id)
            self.cells[neg_cell_id] = cell_negative

        if (cell_positive.dim() == 3):
            if self.debug_export:
                self.complexExporter.write_cell(os.path.join(self.debug_export, "construct_cells"), cell_positive,
                                                count=str(self.cell_count + 1) + "p")
            dd = {"plane_ids": np.array(right_planes)}
            self.cell_count += 1
            pos_cell_id = self.cell_count
            self.tree.create_node(tag=pos_cell_id, identifier=pos_cell_id, data=dd, parent=parent)
            self.graph.add_node(pos_cell_id)
            self.cells[pos_cell_id] = cell_positive


        ## add the split plane to the parent node of the tree
        self.tree.nodes[parent].data["supporting_plane_id"] = best_plane_id_input

        if (cell_positive.dim() == 3 and cell_negative.dim() == 3):

            new_intersection = cell_negative.intersection(cell_positive)
            self.graph.add_edge(neg_cell_id, pos_cell_id, intersection=new_intersection, vertices=[],
                                group_id=id,
                                supporting_plane_id=best_plane_id_input, bounding_box_edge=False)
            if self.debug_export:
                self.complexExporter.write_facet(os.path.join(self.debug_export, "construct_facets"), new_intersection,
                                                 count=self.plane_count)



        ## add edges to other cells, these must be neigbors of the parent (her named child) of the new subspaces
        neighbors_of_old_cell = list(self.graph[parent])
        old_cell_id = parent
        for neighbor_id_old_cell in neighbors_of_old_cell:
            # self.logger.debug("make neighbors")

            # get the neighboring convex
            nconvex = self.cells.get(neighbor_id_old_cell)
            # intersect new cells with old neighbors to make the new facets
            n_nonempty = False;
            p_nonempty = False
            if cell_negative.dim() == 3:
                negative_intersection = nconvex.intersection(cell_negative)
                n_nonempty = negative_intersection.dim() == 2
            if cell_positive.dim() == 3:
                positive_intersection = nconvex.intersection(cell_positive)
                p_nonempty = positive_intersection.dim() == 2
            # add the new edges (from new cells with intersection of old neighbors) and move over the old additional vertices to the new
            if n_nonempty:
                self.graph.add_edge(neighbor_id_old_cell, neg_cell_id, intersection=negative_intersection, vertices=[],
                                    group_id=self.graph[neighbor_id_old_cell][old_cell_id]["group_id"],
                                    supporting_plane_id=self.graph[neighbor_id_old_cell][old_cell_id][
                                        "supporting_plane_id"], convex_intersection=False, bounding_box_edge=False)
            if p_nonempty:
                self.graph.add_edge(neighbor_id_old_cell, pos_cell_id, intersection=positive_intersection, vertices=[],
                                    group_id=self.graph[neighbor_id_old_cell][old_cell_id]["group_id"],
                                    supporting_plane_id=self.graph[neighbor_id_old_cell][old_cell_id][
                                        "supporting_plane_id"], convex_intersection=False, bounding_box_edge=False)

        self.graph.remove_node(parent)
        del self.cells[parent]
        
        return 1


    def add_subdivision(self,res=[2,2,2]):

        self.logger.info("Apply subdivision of input into {} grid".format(res))

        # TODO: when making a grid; just make one with twice the density and sample points on that;
        # then just run this normally until additional planes are all inserted
        def make_grid(res):

            bb = np.array(self.bounding_poly.bounding_box())
            bb_mean = bb.mean(axis=0)

            subdivisions = []
            points = []
            dims = np.array([0,1,2])
            for dim in range(3):
                subdivisions.append(np.linspace(bb[0, dim], bb[1, dim], res[dim] + 1)[1:-1])

                # now I just make sampling points with res = res+1, which automatically ensures that I have a point in each subspace
                tdims = np.delete(dims,dim)
                a = np.linspace(bb[0, tdims[0]], bb[1, tdims[0]], res[tdims[0]] + 2)[1:-1]
                b = np.linspace(bb[0, tdims[1]], bb[1, tdims[1]], res[tdims[1]] + 2)[1:-1]
                # A, B = np.mgrid[a,b]
                A, B = np.meshgrid(a, b)
                points.append(np.vstack([A.ravel(), B.ravel()]).transpose())

            normals = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            normals = np.array(normals)
            planes = []

            ppoints = []
            pnormals = []
            projected_points = []
            for dim in range(3):

                for sub in subdivisions[dim]:
                    n = normals[dim]
                    pt = bb_mean
                    pt[dim] = sub
                    plane = [n[0], n[1], n[2], -np.dot(n, pt)]
                    planes.append(plane)

                    pts = points[dim]
                    projected_points.append(pts)
                    isub = np.repeat(sub, len(pts))
                    pts = np.insert(pts, dim, isub, axis=1)
                    ppoints.append(pts)
                    pnormals.append(np.repeat(n[np.newaxis, :], len(pts), axis=0))

            return np.array(planes), np.array(ppoints), np.array(pnormals), np.array(projected_points)



        planes, points, normals, projected_points = make_grid(res=res)
        color = np.zeros(shape=(1,3))+[255,255,255]

        # add the planes to the vertex_group
        for i,plane in enumerate(planes):
            self.vg.plane_order = np.append(0,self.vg.plane_order+1).astype(self.vg.plane_order.dtype)
            self.vg.input_planes = np.append(plane[np.newaxis,:].astype(self.vg.input_planes.dtype),self.vg.input_planes,axis=0)
            hspace_neg, hspace_pos = self._inequalities(plane)
            self.vg.halfspaces = [np.array([Polyhedron(ieqs=[hspace_neg]),Polyhedron(ieqs=[hspace_pos])])] + self.vg.halfspaces
            self.vg.polygons_from_plane = [np.array([])]+self.vg.polygons_from_plane
            self.vg.hull_vertices = np.append(np.zeros(shape=(1,self.vg.hull_vertices.shape[1]),dtype=self.vg.hull_vertices.dtype),self.vg.hull_vertices,axis=0)
            self.vg.plane_colors = np.append(color.astype(self.vg.plane_colors.dtype),self.vg.plane_colors,axis=0)

            ## add point stuff
            clen = len(self.vg.points)
            self.vg.groups = [np.arange(clen,clen+len(points[i]),dtype=self.vg.groups[0].dtype)] + self.vg.groups
            self.vg.points = np.append(self.vg.points,points[i].astype(self.vg.points.dtype),axis=0)
            self.vg.normals = np.append(self.vg.normals,normals[i].astype(self.vg.points.dtype),axis=0)
            self.vg.classes = np.append(self.vg.classes,np.zeros(clen).astype(self.vg.classes.dtype))
            self.vg.projected_points = np.append(self.vg.projected_points,projected_points[i].astype(self.vg.projected_points.dtype),axis=0)

        if not self.partition_initialized:
            self._init_partition()

        children = self.tree.expand_tree(0, filter=lambda x: x.data["plane_ids"].shape[0], mode=self.tree_mode)
        for child in children:



            # current_ids = self.tree[child].data["additional_splits"]
            current_ids = self.tree[child].data["plane_ids"]
            current_cell = self.cells.get(child)

            best_plane_id = 0
            best_plane_id_input = self.vg.plane_ids[current_ids[best_plane_id]]
            if best_plane_id_input >= len(planes):
                continue

            best_plane = self.vg.split_planes[current_ids[best_plane_id]]
            ### split the point sets with the best_plane, and append the split sets to the self.vg arrays
            left_planes, right_planes = self._split_support_points(best_plane_id, current_ids)



            ### for debug export
            self.best_plane_ids.append(best_plane_id_input)
            self.plane_count+=1
            ### export best plane
            if self.debug_export:
                epoints = self.vg.points[self.vg.groups[current_ids[best_plane_id]]]
                color = self.vg.plane_colors[best_plane_id_input]
                if len(epoints) > 3:
                    self.planeExporter.save_plane(os.path.join(self.debug_export,"additional_planes"), best_plane, epoints, count=str(self.plane_count), color=color)

            ## insert the best plane into the complex
            self._insert_new_plane(parent=child,id=current_ids[best_plane_id],left_planes=left_planes,right_planes=right_planes)

            ## progress bar update
            n_points_processed = len(self.vg.groups[current_ids[best_plane_id]])

        self.polygons_initialized = False # false because I do not initialize the sibling facets

        return 0




    def _init_partition(self):

        self.vg.plane_ids = list(range(self.vg.input_planes.shape[0]))
        self.vg.split_planes = copy.deepcopy(self.vg.input_planes)

        self.cell_count = 0
        self.split_count = 0

        ## init the graph
        self.graph = nx.Graph()
        self.graph.add_node(self.cell_count, convex=self.bounding_poly)

        ## expand the tree as long as there is at least one plane inside any of the subspaces
        self.tree = Tree()
        dd = {"plane_ids": np.arange(self.vg.split_planes.shape[0])}
        self.tree.create_node(tag=self.cell_count, identifier=self.cell_count, data=dd)  # root node
        self.cells[self.cell_count] = self.bounding_poly
        self.plane_count = 0 # only used for debugging exports
        self.best_plane_ids = [] # only used for debugging exports

        self.partition_initialized = True

    def _compute_split(self, parent, insertion_order="product-earlystop"):

        disable_tqdm = False if self.verbosity < 30 else True
        pbar = tqdm(total=self.n_points,file=sys.stdout, disable=disable_tqdm)

        children = self.tree.expand_tree(parent, filter=lambda x: x.data["plane_ids"].shape[0], mode=self.tree_mode)
        for child in children:

            current_ids = self.tree[child].data["plane_ids"]
            current_cell = self.cells.get(child)

            if current_cell is None:  # necessary for subdivision schemes
                continue

            ## get the best plane              
            if len(current_ids) == 1:
                best_plane_id = 0
                best_plane = self.vg.split_planes[current_ids[best_plane_id]]
                left_planes = [];
                right_planes = []
            else:
                best_plane_id = 0 if not insertion_order else self._get_best_plane(current_ids, insertion_order)
                best_plane = self.vg.split_planes[current_ids[best_plane_id]]
                ### split the point sets with the best_plane, and append the split sets to the self.vg arrays
                left_planes, right_planes = self._split_support_points(best_plane_id, current_ids)

            ### for debug export
            best_plane_id_input = self.vg.plane_ids[current_ids[best_plane_id]]
            self.best_plane_ids.append(best_plane_id_input)
            self.plane_count += 1
            ### export best plane
            if self.debug_export:
                epoints = self.vg.points[self.vg.groups[current_ids[best_plane_id]]]
                color = self.vg.plane_colors[best_plane_id_input]
                if len(epoints) > 3:
                    self.planeExporter.save_plane(os.path.join(self.debug_export, "split_planes"), best_plane, epoints,
                                                  count=str(self.plane_count), color=color)

            ## insert the best plane into the complex
            self._insert_new_plane(parent=child, id=current_ids[best_plane_id], left_planes=left_planes,
                                   right_planes=right_planes)

            ## progress bar update
            n_points_processed = len(self.vg.groups[current_ids[best_plane_id]])
            pbar.update(n_points_processed)

        pbar.close()



    def construct_partition(self, insertion_order="product-earlystop"):
        """
        1. Construct the partition
        :param insertion_order: In which order to process the planes.
        """
        self.logger.info('Construct partition with mode {} on {}'.format(insertion_order, self.device))

        if not self.partition_initialized:
            self._init_partition()

        self._compute_split(0, insertion_order=insertion_order)

        self.polygons_initialized = False # false because I do not initialize the sibling facets
        self.logger.debug("Plane insertion order {}".format(self.best_plane_ids))
        self.logger.debug("{} input planes were split {} times, making a total of {} planes now".format(len(self.vg.input_planes),self.split_count,len(self.vg.split_planes)))


    def save_partition_to_pickle(self,outpath):

        self.logger.info("Save partition to pickle...")
        self.logger.debug("to {}".format(outpath))

        os.makedirs(outpath,exist_ok=True)

        if self.tree is not None:
            pickle.dump(self.tree,open(os.path.join(outpath,'tree.pickle'),'wb'))
        pickle.dump(self.graph,open(os.path.join(outpath,'graph.pickle'),'wb'))
        pickle.dump(self.cells,open(os.path.join(outpath,'cells.pickle'),'wb'))
        pickle.dump(self.vg.groups,open(os.path.join(outpath,'groups.pickle'),'wb'))
        pickle.dump(self.vg.input_planes,open(os.path.join(outpath,'planes.pickle'),'wb'))




    def load_partition_from_pickle(self,inpath):

        self.logger.info("Load partition from pickle...")

        self.tree = pickle.load(open(os.path.join(inpath,'tree.pickle'),'rb'))
        self.graph = pickle.load(open(os.path.join(inpath,'graph.pickle'),'rb'))
        self.cells = pickle.load(open(os.path.join(inpath,'cells.pickle'),'rb'))
        self.vg.groups = pickle.load(open(os.path.join(inpath,'groups.pickle'),'rb'))
        self.vg.input_planes = pickle.load(open(os.path.join(inpath,'planes.pickle'),'rb'))

        assert(len(self.cells) == len(self.graph.nodes)) ## this makes sure that every graph node has a convex attached

        self.polygons_initialized = False # false because I do not initialize the sibling facets


    def construct_abspy(self, exhaustive=False, num_workers=0):
        """
        Construct cell complex.

        Two-stage primitive-in-cell predicate.
        (1) bounding boxes of primitive and existing cells are evaluated
        for possible intersection. (2), a strict intersection test is performed.

        Generated cells are stored in self.cells.
        * query the bounding box intersection.
        * optional: intersection test for polygon and edge in each potential cell.
        * partition the potential cell into two. rewind if partition fails.

        Parameters
        ----------
        exhaustive: bool
            Do exhaustive partitioning if set True
        num_workers: int
            Number of workers for multi-processing, disabled if set 0
        """

        def _intersect_neighbour(kwargs):
            """
            Intersection test between partitioned cells and neighbouring cell.
            Implemented for multi-processing across all neighbours.

            Parameters
            ----------
            kwargs: (int, Polyhedron object, Polyhedron object, Polyhedron object)
                (neighbour index, positive cell, negative cell, neighbouring cell)
            """
            n, cell_positive, cell_negative, cell_neighbour = kwargs['n'], kwargs['positive'], kwargs['negative'], \
                                                              kwargs['neighbour']

            interface_positive = cell_positive.intersection(cell_neighbour)

            if interface_positive.dim() == 2:
                # this neighbour can connect with either or both children
                self.graph.add_edge(self.index_node + 1, n, supporting_plane_id=kwargs["supporting_plane_id"])
                interface_negative = cell_negative.intersection(cell_neighbour)
                if interface_negative.dim() == 2:
                    self.graph.add_edge(self.index_node + 2, n, supporting_plane_id=kwargs["supporting_plane_id"])
            else:
                # this neighbour must otherwise connect with the other child
                self.graph.add_edge(self.index_node + 2, n, supporting_plane_id=kwargs["supporting_plane_id"])


        def _index_node_to_cell(query):
            """
            Convert index in the node list to that in the cell list.
            The rationale behind is #nodes == #cells (when a primitive is settled down).

            Parameters
            ----------
            query: int
                Query index in the node list

            Returns
            -------
            as_int: int
                Query index in the cell list
            """
            return list(self.graph.nodes).index(query)

        def _intersect_bound_plane(bound, plane, exhaustive=False, epsilon=10e-5):
            """
            Pre-intersection test between query primitive and existing cells,
            based on AABB and plane parameters.

            Parameters
            ----------
            bound: (2, 3) float
                Bound of the query planar primitive
            plane: (4,) float
                Plane parameters
            exhaustive: bool
                Exhaustive partitioning, only for benchmarking
            epsilon: float
                Distance tolerance

            Returns
            -------
            as_int: (n,) int
                Indices of existing cells whose bounds intersect with bounds of the query primitive
                and intersect with the supporting plane of the primitive
            """
            if exhaustive:
                return np.arange(len(self.cells_bounds))

            # each planar primitive partitions only the 3D cells that intersect with it
            cells_bounds = np.array(self.cells_bounds)  # easier array manipulation
            center_targets = np.mean(cells_bounds, axis=1)  # N * 3
            extent_targets = cells_bounds[:, 1, :] - cells_bounds[:, 0, :]  # N * 3

            if bound[0][0] == -np.inf:
                intersection_bound = np.arange(len(self.cells_bounds))

            else:
                # intersection with existing cells' AABB
                center_query = np.mean(bound, axis=0)  # 3,
                center_distance = np.abs(center_query - center_targets)  # N * 3
                extent_query = bound[1] - bound[0]  # 3,

                # abs(center_distance) * 2 < (query extent + target extent) for every dimension -> intersection
                intersection_bound = \
                np.where(np.all(center_distance * 2 < extent_query + extent_targets + epsilon, axis=1))[0]

            # plane-AABB intersection test from extracted intersection_bound only
            # https://gdbooks.gitbooks.io/3dcollisions/content/Chapter2/static_aabb_plane.html
            # compute the projection interval radius of AABB onto L(t) = center + t * normal
            radius = np.dot(extent_targets[intersection_bound] / 2, np.abs(plane[:3]))
            # compute distance of box center from plane
            distance = np.dot(center_targets[intersection_bound], plane[:3]) + plane[3]
            # intersection between plane and AABB occurs when distance falls within [-radius, +radius] interval
            intersection_plane = np.where(np.abs(distance) <= radius + epsilon)[0]

            return intersection_bound[intersection_plane]

        self.graph = nx.Graph()
        self.graph.add_node(0)  # the initial cell
        self.index_node = 0  # unique for every cell ever generated

        self.cells_bounds = [self.bounding_poly.bounding_box()]
        self.cells = [self.bounding_poly]
        cell_dict = dict()
        cell_dict[self.index_node] = self.bounding_poly

        if exhaustive:
            self.logger.info('construct exhaustive cell complex'.format())
        else:
            self.logger.info('construct cell complex'.format())

        tik = time.time()

        pool = None
        if num_workers > 0:
            pool = multiprocessing.Pool(processes=num_workers)

        pbar = trange(len(self.vg.bounds),file=sys.stdout)
        for i in pbar:  # kinetic for each primitive
            # bounding box intersection test
            # indices of existing cells with potential intersections
            indices_cells = _intersect_bound_plane(self.vg.bounds[i], self.vg.split_planes[i], exhaustive)
            assert len(indices_cells), 'intersection failed! check the initial bound'

            # half-spaces defined by inequalities
            # no change_ring() here (instead, QQ() in _inequalities) speeds up 10x
            # init before the loop could possibly speed up a bit
            hspace_positive, hspace_negative = [Polyhedron(ieqs=[inequality]) for inequality in
                                                self._inequalities(self.vg.split_planes[i])]


            # partition the intersected cells and their bounds while doing mesh slice plane
            indices_parents = []

            for index_cell in indices_cells:
                cell_positive = hspace_positive.intersection(self.cells[index_cell])
                cell_negative = hspace_negative.intersection(self.cells[index_cell])

                if cell_positive.dim() != 3 or cell_negative.dim() != 3:
                    # if cell_positive.is_empty() or cell_negative.is_empty():
                    """
                    cannot use is_empty() predicate for degenerate cases:
                        sage: Polyhedron(vertices=[[0, 1, 2]])
                        A 0-dimensional polyhedron in ZZ^3 defined as the convex hull of 1 vertex
                        sage: Polyhedron(vertices=[[0, 1, 2]]).is_empty()
                        False
                    """
                    continue

                # incrementally build the adjacency graph
                if self.graph is not None:
                    # append the two nodes (UID) being partitioned
                    self.graph.add_node(self.index_node + 1)
                    self.graph.add_node(self.index_node + 2)
                    cell_dict[self.index_node+1] = cell_positive
                    cell_dict[self.index_node+2] = cell_negative

                    # append the edge in between
                    self.graph.add_edge(self.index_node + 1, self.index_node + 2,supporting_plane_id=i)

                    # get neighbours of the current cell from the graph
                    neighbours = self.graph[list(self.graph.nodes)[index_cell]]  # index in the node list

                    if neighbours:
                        # get the neighbouring cells to the parent
                        cells_neighbours = [self.cells[_index_node_to_cell(n)] for n in neighbours]

                        kwargs = []
                        for n, cell in zip(neighbours, cells_neighbours):
                            supporting_plane_id = self.graph.edges[list(self.graph.nodes)[index_cell],n]["supporting_plane_id"]
                            kwargs.append({'n': n, 'positive': cell_positive, 'negative': cell_negative, 'neighbour': cell,
                                           'supporting_plane_id':supporting_plane_id})

                        if pool is None:
                            for k in kwargs:
                                _intersect_neighbour(k)
                        else:
                            pool.map(_intersect_neighbour, kwargs)

                    # update cell id
                    self.index_node += 2

                self.cells.append(cell_positive)
                self.cells.append(cell_negative)


                # incrementally cache the bounds for created cells
                self.cells_bounds.append(cell_positive.bounding_box())
                self.cells_bounds.append(cell_negative.bounding_box())

                indices_parents.append(index_cell)

            # delete the parent cells and their bounds. this does not affect the appended ones
            for index_parent in sorted(indices_parents, reverse=True):
                del self.cells[index_parent]
                del self.cells_bounds[index_parent]

                # remove the parent node (and subsequently its incident edges) in the graph
                if self.graph is not None:
                    del cell_dict[list(self.graph.nodes)[index_parent]]
                    self.graph.remove_node(list(self.graph.nodes)[index_parent])

        self.logger.debug('cell complex constructed: {:.2f} s'.format(time.time() - tik))

        self.cells=cell_dict


