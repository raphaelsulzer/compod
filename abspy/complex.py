"""
complex.py
----------

Cell complex from planar primitive arrangement.

A linear cell complex is constructed from planar primitives
with adaptive binary space partitioning: upon insertion of a primitive
only the local cells that are intersecting it will be updated,
so will be the corresponding adjacency graph of the complex.
"""

from .setup import *
import os, sys, time, multiprocessing, pickle
from pathlib import Path
from random import random
from fractions import Fraction
import numpy as np
from tqdm import trange
import networkx as nx
from sage.all import QQ, RR, Polyhedron, vector, arctan2
from treelib import Tree
from tqdm import tqdm
import open3d as o3d

from .logger import attach_to_log
logger = attach_to_log()

from .export_complex import CellComplexExporter
import libPyLabeler as PL
import libSoup2Mesh as s2m
from export import PlaneExporter
from pyplane import PyPlane, SagePlane, ProjectedConvexHull
from color import FancyColor

class CellComplex:
    """
    Class of cell complex from planar primitive arrangement.
    """
    def __init__(self, model, vertex_group, initial_padding=0.1, export=False, device='cpu'):
        """
        Init CellComplex.
        Class of cell complex from planar primitive arrangement.

        Parameters
        ----------
        planes: (n, 4) float
            Plana parameters
        bounds: (n, 2, 3) float
            Corresponding bounding box bounds of the planar primitives
        points: (n, ) object of float
            Points grouped into primitives, points[any]: (m, 3)
        initial_bound: None or (2, 3) float
            Initial bound to partition
        build_graph: bool
            Build the cell adjacency graph if set True.
        additional_planes: None or (n, 4) float
            Additional planes to append to the complex,
            can be missing planes due to occlusion or incapacity of RANSAC
        quiet: bool
            Disable logging and progress bar if set True
        """
        self.model = model
        self.planeExporter = PlaneExporter()
        self.cellComplexExporter = CellComplexExporter(self)

        logger.debug('Init cell complex with padding {}'.format(initial_padding))

        vertex_group.planes, vertex_group.halfspaces, vertex_group.bounds, vertex_group.points_grouped,
        self.bounds = vertex_group.bounds
        self.planes = vertex_group.planes
        self.plane_order = vertex_group.plane_order
        self.plane_dict = vertex_group.plane_dict
        self.plane_colors = vertex_group.plane_colors
        self.halfspaces = vertex_group.halfspaces
        self.points = vertex_group.points_grouped
        self.hull_vertices = vertex_group.hull_vertices
        self.convex_hulls = vertex_group.convex_hulls
        self.vertex_group_n_fill = vertex_group.n_fill
        del vertex_group

        self.cells = dict()
        self.tree = None
        self.graph = None
        self.device = device
        if self.device == 'gpu':
            import torch
            self.torch = torch
        else:
            self.torch = None

        self.polygons_initialized = False

        # init the bounding box
        self.initial_padding = initial_padding
        self.bounding_poly = self._init_bounding_box(padding=self.initial_padding)

        self.export = export


    def _init_bounding_box(self,padding):

        self.bounding_verts = []

        points = np.concatenate(self.points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        aabb = pcd.get_axis_aligned_bounding_box()
        # aabb.scale(padding,aabb.get_center())
        ppmin = aabb.min_bound.astype(points.dtype)
        ppmax = aabb.max_bound.astype(points.dtype)

        ### very important to have the same types, otherwise the bounding planes will not be correctly inserted
        assert(points.dtype == np.float32)
        assert(ppmin.dtype == np.float32)

        pmin=[]
        for p in ppmin:
            pmin.append(Fraction(str(p)))
        pmax=[]
        for p in ppmax:
            pmax.append(Fraction(str(p)))


        pmin = vector(pmin)
        pmax = vector(pmax)
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

    def _sort_vertex_indices_by_angle(self,points,plane):
        '''order vertices of a convex polygon:
        https://blogs.sas.com/content/iml/2021/11/17/order-vertices-convex-polygon.html#:~:text=Order%20vertices%20of%20a%20convex%20polygon&text=You%20can%20use%20the%20centroid,vertices%20of%20the%20convex%20polygon
        '''
        ## project to plane
        # pp=PyPlane(plane).project_points_to_plane_coordinate_system(points)
        pp=PyPlane(plane).to_2d(points)

        center = np.mean(pp,axis=0)
        vectors = pp - center
        # vectors = vectors/np.linalg.norm(vectors)
        radians = np.arctan2(vectors[:,1],vectors[:,0])

        if (np.unique(radians,return_counts=True)[1]>1).any():
            a=5


        # same_rads = radians[np.unique(radians,return_counts=True)[1]>1]
        # if same_rads.shape[0]:
        #     print("WARNING: same angle")
        #     return None

        return np.argsort(radians)

    def _orient_exact_polygon(self, points, outside):
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

        return dot < 0

    def _orient_inexact_polygon(self, points, outside):
        # check for left or right orientation
        # https://math.stackexchange.com/questions/2675132/how-do-i-determine-whether-the-orientation-of-a-basis-is-positive-or-negative-us

        i = 0
        cross=0
        while np.sum(cross) == 0:
            a = points[i+1] - points[i]
            a = a/np.linalg.norm(a)
            b = points[i+2] - points[i]
            b = b/np.linalg.norm(b)
            cross = np.cross(a,b)
            i+=1

        c = np.array(outside) - points[i]
        c = c/np.linalg.norm(c)
        cross = cross/np.linalg.norm(cross)
        dot = np.dot(cross,c)

        return dot < 0

    def _get_intersection(self, e0, e1):

        if "vertices" in self.graph[e0][e1] and self.graph[e0][e1]["vertices"] is not None:
            pts = []
            for v in self.graph[e0][e1]["vertices"]:
                pts.append(tuple(v))
            pts = list(set(pts))
            intersection_points = np.array(pts, dtype=object)
        elif "intersection" in self.graph[e0][e1] and self.graph[e0][e1] is not None:
            intersection_points = np.array(self.graph[e0][e1]["intersection"].vertices_list(), dtype=object)
        else:
            intersection = self.cells.get(e0).intersection(self.cells.get(e1))
            assert(intersection.dim()==2)
            intersection_points = np.array(intersection.vertices_list(), dtype=object)

        return intersection_points


    def extract_colored_soup(self, filename):

        logger.info('Extract colored soup...')

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

                # TODO: a better solution instead of using a plane dict is simply to get the ID from the primitive_dict["plane_ids"] array
                plane_id = self.plane_dict.get(str(self.graph.edges[e0, e1]["supporting_plane"]), -1)
                col = self.plane_colors[plane_id] if plane_id > -1 else np.random.randint(100, 255, size=3)
                fcolors.append(col)

                intersection_points = self._get_intersection(e0, e1)

                correct_order = self._sort_vertex_indices_by_angle_exact(intersection_points,
                                                                         self.graph[e0][e1]["supporting_plane"])

                assert (len(intersection_points) == len(correct_order))
                intersection_points = intersection_points[correct_order]

                if (len(intersection_points) < 3):
                    print("ERROR: Encountered facet with less than 2 vertices.")
                    sys.exit(1)

                ## orient polygon
                outside = self.cells.get(e0).center() if c1["occupancy"] else self.cells.get(e1).center()
                # if self._orient_inexact_polygon(intersection_points_float,np.array(outside).astype(float)):
                if self._orient_exact_polygon(intersection_points, outside):
                    intersection_points = np.flip(intersection_points, axis=0)

                for pt in intersection_points:
                    all_points.append(pt)
                    pcolors.append(col)
                # for cgal export
                faces.append(np.arange(len(intersection_points)) + n_points)
                n_points += len(intersection_points)

        all_points = np.array(all_points, dtype=float)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        logger.debug('Save colored polygon soup to {}'.format(filename))

        self.cellComplexExporter.write_colored_soup_to_ply(filename, points=all_points,
                                                              facets=faces, pcolors=pcolors,fcolors=fcolors)


    # @profile
    def extract_surface(self, filename, backend = "python", triangulate = False):

        logger.info('Extract surface...')


        tris = []
        colors = []
        all_points = []
        # for cgal export
        faces = []
        face_lens = []
        n_points = 0
        for e0, e1 in self.graph.edges:

            # if e0 > e1:
            #     continue

            c0 = self.graph.nodes[e0]
            c1 = self.graph.nodes[e1]

            if c0["occupancy"] != c1["occupancy"]:

                intersection_points = self._get_intersection(e0,e1)

                correct_order = self._sort_vertex_indices_by_angle_exact(intersection_points,self.graph[e0][e1]["supporting_plane"])

                assert(len(intersection_points)==len(correct_order))
                intersection_points = intersection_points[correct_order]

                if(len(intersection_points)<3):
                    print("ERROR: Encountered facet with less than 2 vertices.")
                    sys.exit(1)

                ## orient polygon
                outside = self.cells.get(e0).center() if c1["occupancy"] else self.cells.get(e1).center()
                # if self._orient_inexact_polygon(intersection_points_float,np.array(outside).astype(float)):
                if self._orient_exact_polygon(intersection_points,outside):
                    intersection_points = np.flip(intersection_points, axis=0)

                for i in range(intersection_points.shape[0]):
                    all_points.append(tuple(intersection_points[i,:]))
                tris.append(intersection_points)
                # for cgal export
                faces.append(np.arange(len(intersection_points))+n_points)
                face_lens.append(len(intersection_points))
                n_points+=len(intersection_points)

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        logger.debug('Save polygon with backend {} mesh to {}'.format(backend,filename))

        if backend == "cgal":
            sm = s2m.Soup2Mesh()
            sm.loadSoup(np.array(all_points,dtype=float), np.array(face_lens,dtype=int), np.concatenate(faces,dtype=int))
            sm.makeMesh(triangulate)
            sm.saveMesh(filename)
        elif backend == "python":
            pset = set(all_points)
            pset = np.array(list(pset),dtype=object)
            facets = []
            for tri in tris:
                face = []
                for pt in tri:
                    face.append(np.argwhere((np.equal(pset,pt,dtype=object)).all(-1))[0][0])
                facets.append(face)

            self.cellComplexExporter.write_surface_to_off(filename,points=np.array(pset,dtype=np.float32),facets=facets)
        else:
            raise NotImplementedError


    def extract_in_cells_explode(self,filename,shrink_percentage=0.01):

        logger.info('Extract inside cells...')

        os.makedirs(os.path.dirname(filename),exist_ok=True)
        f = open(filename,'w')

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
        col=FancyColor(bbox)
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
            c = col.get_rgb_from_xyz(centroid)
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


    def extract_in_cells(self,filename):

        logger.info('Extract inside cells...')

        os.makedirs(os.path.dirname(filename),exist_ok=True)
        f = open(filename,'w')

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


    def extract_partition_as_ply(self, filepath, rand_colors=True, export_boundary=True, with_primitive_id=True):
        logger.info('Extract partition...')

        """
        Save polygon soup of indexed convexes to an obj file.

        Parameters
        ----------
        filepath: str or Path
            Filepath to save obj file
        indices_cells: (n,) int
            Indices of cells to save to file
        use_mtl: bool
            Use mtl attribute in obj if set True
        """
        # create the dir if not exists
        if not self.polygons_initialized:
            self._init_polygons()


        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        points = []
        indices = []
        primitive_ids = []
        count = 0
        supporting_planes = []
        ecount = 0

        colors = []
        for c0,c1 in self.graph.edges:


            if not export_boundary:
                bbe = self.graph.edges[c0,c1].get("bounding_box_edge",False)
                if bbe:
                    continue


            ecount+=1
            face = self.graph.edges[c0,c1]["intersection"]
            correct_vertex_order = self._sorted_vertex_indices(face.adjacency_matrix())
            points.append(np.array(face.vertices_list())[correct_vertex_order])
            indices.append(list(np.arange(count,len(correct_vertex_order)+count)))


            plane = self.graph.edges[c0,c1]["supporting_plane"]
            supporting_planes.append(plane)
            ## if plane_id is negative append negative primitive_id

            # TODO: a better solution instead of using a plane dict is simply to get the ID from the primitive_dict["plane_ids"] array
            if str(plane) in self.plane_dict.keys():
                plane_id = self.plane_dict[str(plane)]
            else:
                logger.warning("\nPlane not found in plane_dict")
                plane_id = -1

            if plane_id > -1:
                colors.append(self.plane_colors[plane_id])
                primitive_ids.append([self.plane_order[plane_id]])
            else:
                colors.append(np.random.randint(100, 255, size=3))
                primitive_ids.append([])


            count+=len(correct_vertex_order)

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


    def graph_cut(self):

        logger.info("Apply label regularization with graph-cut...")

        dtype = np.int64
        # Internally, in the pyGCO code datatype is always converted to np.int32
        # I would need to use my own version of GCO (or modify the one used by pyGCO) to change that
        # Should probably be done at some point to avoid int32 overflow for larger scenes.



        gc = gco.GCO()
        gc.create_general_graph(len(self.graph.nodes) + 1, 2, energy_is_float=False)
        # data_cost = F.softmax(prediction, dim=-1)
        prediction[:, [0, 1]] = prediction[:, [1, 0]]
        data_cost = (prediction * clf.graph_cut.unary_weight).round()
        # data_cost = prediction
        data_cost = np.array(data_cost, dtype=dtype)
        ### append high cost for inside for infinite cell
        # data_cost = np.append(data_cost, np.array([[-10000, 10000]]), axis=0)
        gc.set_data_cost(data_cost)
        smooth = (1 - np.eye(2)).astype(dtype)
        gc.set_smooth_cost(smooth)
        if (not clf.graph_cut.binary_type):
            edge_weight = np.ones(edges.shape[0], dtype=dtype)
        else:
            # TODO: retrieve the beta-skeleton and area value from the features to weight the binaries
            edge_weight = np.ones(edges.shape[0], dtype=dtype)

        gc.set_all_neighbors(edges[:, 0], edges[:, 1], edge_weight * clf.graph_cut.binary_weight)

        for i, l in enumerate(labels):
            gc.init_label_at_site(i, l)

        # print("before smooth: ",gc.compute_smooth_energy())
        # print("before data: ", gc.compute_data_energy())
        # print("before: ", gc.compute_smooth_energy()+gc.compute_data_energy())

        gc.expansion()

        # print("after: ", gc.compute_smooth_energy()+gc.compute_data_energy())

        # print("after smooth: ",gc.compute_smooth_energy())
        # print("after data: ", gc.compute_data_energy())

        labels = gc.get_labels()

        return labels


    def label_cells(self, m, n_test_points=50,graph_cut=True,export=False):

        pl=PL.PyLabeler(n_test_points)
        if pl.loadMesh(m["mesh"]):
            return 1
        points = []
        points_len = []
        for i,id in enumerate(list(self.graph.nodes)):
            if id < 0:  continue # skip the bounding box cells
            cell = self.cells.get(id)
            if export:
                self.cellComplexExporter.write_cell(m,cell,count=i,subfolder="final_cells")
            pts = np.array(cell.vertices())
            points.append(pts)
            # print(pts)
            points_len.append(pts.shape[0])


        # assert(isinstance(points[0].dtype,np.float32))
        occs = pl.labelCells(np.array(points_len),np.concatenate(points,axis=0))
        del pl

        foccs = dict(zip(self.graph.nodes, occs))
        nx.set_node_attributes(self.graph,occs,"float_occupancy")

        if graph_cut:
            # occs = self.graph_cut(occs)
            occs = dict(zip(self.graph.nodes, np.rint(occs).astype(int)))
        else:
            occs = dict(zip(self.graph.nodes, np.rint(occs).astype(int)))

        nx.set_node_attributes(self.graph,occs,"occupancy")



    @profile
    def _get_best_split(self,current_ids,primitive_dict,insertion_order="product-earlystop"):

        ## sum calls the get_best_plane function less often than product, but with more planes
        ## since get_best_planes is O(n^2), it is better to call it more often with smaller n than less often with bigger n

        earlystop = False
        if "earlystop" in insertion_order:
            earlystop = True

        planes = primitive_dict["planes"][current_ids]
        point_groups = np.array(primitive_dict["point_groups"],dtype=object)[current_ids]


        ### find the plane which seperates all other planes without splitting them
        left_right = []
        for i,id in enumerate(current_ids):
            left = 0; right = 0; intersect = 0
            pls = planes[i]
            for j,id2 in enumerate(current_ids):
                if i == j: continue
                which_side = np.dot(pls[:3],point_groups[j].transpose())
                which_side = (which_side < -pls[3])
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
            left_right = np.product(left_right[:,:2],axis=1)
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


    @profile
    def _get_best_split_gpu(self,current_ids,primitive_dict,insertion_order="product-earlystop"):

        # TODO: in fact left_right has to be computed only one single time per plane and can be stored throughout the algorithm
        # a simple first try could be to store it for each primitve and simply make a function which sorts it at each insertion according to the current available cell ids


        ## sum calls the get_best_plane function less often than product, but with more planes
        ## since get_best_planes is O(n^2), it is better to call it more often with smaller n than less often with bigger n



        ### pad the point groups with NaNs to make a numpy array from the variable lenght list
        ### could maybe better be done with scipy sparse, but would require to rewrite the _get and _split functions used below

        ### the whole thing vectorized. doesn't really work for some reason
        # UPDATE: should work, first tries where with wrong condition
        # planes = np.repeat(vertex_group.planes[np.newaxis,current_ids,:],current_ids.shape[0],axis=0)
        # pgs = np.repeat(point_groups[np.newaxis,current_ids,:,:],current_ids.shape[0],axis=0)
        #
        # which_side = planes[:,:,0,np.newaxis] * pgs[:,:,:,0] + planes[:,:,1,np.newaxis] * pgs[:,:,:,1] + planes[:,:,2,np.newaxis] * pgs[:,:,:,2] + planes[:,:,3,np.newaxis]

        earlystop = False
        if "earlystop" in insertion_order:
            earlystop = True

        planes = primitive_dict["planes"][current_ids]
        hull_verts = primitive_dict['hull_vertices'][current_ids]
        planes = self.torch.from_numpy(planes).type(self.torch.float32).to('cuda')

        ### find the plane which seperates all other planes without splitting them
        left_right = []
        for i,id in enumerate(current_ids):
            pls = planes[i]

            pv = hull_verts.transpose(2,0)
            pv = self.torch.cat((pv[:,:,:i],pv[:,:,i+1:]),axis=2)

            which_side = self.torch.tensordot(pls[:3], pv, 1)
            which_side = which_side < -pls[3]
            left = which_side.all(axis=0)
            right = (~which_side).all(axis=0)

            lr = self.torch.logical_or(left,right)

            if earlystop and lr.all():
                logger.debug("earlystop")
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
            left_right = np.product(left_right[:,:2],axis=1)
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


    def _get_best_plane(self,current_ids,primitive_dict,insertion_order="product-earlystop"):

        """
        Prioritise certain planes to favour building reconstruction.

        First, vertical planar primitives are accorded higher priority than horizontal or oblique ones
        to avoid incomplete partitioning due to missing data about building facades.
        Second, in the same priority class, planar primitives with larger areas are assigned higher priority
        than smaller ones, to make the final cell complex as compact as possible.
        Note that this priority setting is designed exclusively for building models.

        Parameters
        ----------
        prioritise_verticals: bool
            Prioritise vertical planes if set True
        """

        # pgs = np.array(primitive_dict["point_groups"], dtype=object)[current_ids]
        pgs = None

        if insertion_order == "random":
            return np.random.choice(len(current_ids),size=1)[0]
        elif insertion_order in ["product", "product-earlystop", "sum", "sum-earlystop", "equal", "equal-earlystop", "intersect", "intersect-earlystop"]:
            if self.device == "gpu":
                return self._get_best_split_gpu(current_ids,primitive_dict,insertion_order)
            elif self.device == "cpu":
                return self._get_best_split(current_ids,primitive_dict,insertion_order)
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
            for i,plane in enumerate(primitive_dict["planes"][current_ids]):
                if pgs[i].shape[0] > 2:
                    mesh = PyPlane(plane).get_trimesh_of_projected_points(pgs[i])
                    areas.append(mesh.area)
                else:
                    areas.append(0)
            return np.argmax(areas)
        else:
            raise NotImplementedError

    @profile
    def _split_support_points_gpu(self,best_plane_id,current_ids,primitive_dict, th=1):

        '''
        :param best_plane_id:
        :param current_ids:
        :param planes:
        :param point_groups: padded 2d array of point groups with NaNs
        :param n_points_per_plane: real number of points per group (ie plane)
        :return: left and right planes
        '''

        assert th >= 0,"Threshold must be >= 0"

        best_plane = primitive_dict["planes"][current_ids[best_plane_id]]



        ### now put the planes into the left and right subspace of the best_plane split
        ### planes that lie in both subspaces are split (ie their point_groups are split) and appended as new planes to the planes array, and added to both subspaces
        left_plane_ids = []
        right_plane_ids = []
        for id in current_ids:

            if id == current_ids[best_plane_id]:
                continue

            # polygon_points = primitive_dict["point_groups"][id]
            polygon_points = primitive_dict["convex_hulls"][id].all_points

            which_side = np.dot(best_plane[:3],polygon_points.transpose())
            left_points = polygon_points[which_side < -best_plane[3], :]
            right_points = polygon_points[which_side > -best_plane[3], :]

            if (polygon_points.shape[0] - left_points.shape[0]) <= th:
                left_plane_ids.append(id)
            elif(polygon_points.shape[0] - right_points.shape[0]) <= th:
                right_plane_ids.append(id)
            else:
                # print("id:{}: total-left/right: {}-{}/{}".format(current_ids[best_plane_id],n_points_per_plane[id],left_points.shape[0],right_points.shape[0]))
                if (left_points.shape[0] > th):
                    left_plane_ids.append(primitive_dict["planes"].shape[0])
                    primitive_dict["planes"] = np.vstack((primitive_dict["planes"], primitive_dict["planes"][id]))
                    primitive_dict["plane_ids"].append(primitive_dict["plane_ids"][id])
                    primitive_dict["halfspaces"].append(primitive_dict["halfspaces"][id])
                    primitive_dict["split_count"].append(primitive_dict["split_count"][id]+1)
                    primitive_dict["point_groups"].append(left_points)
                    
                    # if not enough points for making a convex hull we simply keep the points 
                    
                    # get all hull points and  make a new hull on the left side
                    if left_points.shape[0] > 2:
                        new_hull = ProjectedConvexHull(primitive_dict["convex_hulls"][id].plane_params,left_points)
                    else:
                        new_hull = primitive_dict["convex_hulls"][id]
                        new_hull.hull = None
                        new_hull.hull_points = left_points
                        new_hull.all_points = left_points
                    primitive_dict["convex_hulls"].append(new_hull)
                    hull_points = new_hull.hull_points

                    if hull_points.shape[0] <= self.vertex_group_n_fill:
                        fill = self.vertex_group_n_fill - hull_points.shape[0]
                        fill = hull_points[np.random.choice(hull_points.shape[0],fill)]
                        hull_points=np.concatenate((hull_points,fill))
                    else:
                        logger.warning("Fill value overflow. Len of hull points = {}, fill value = {}".format(hull_points.shape[0],self.vertex_group_n_fill))
                        hull_points = hull_points[:self.vertex_group_n_fill]
                    primitive_dict["hull_vertices"] = self.torch.cat(
                        (primitive_dict["hull_vertices"],
                         self.torch.Tensor(np.array(hull_points)).to('cuda')[None, :, :]))

                if (right_points.shape[0] > th):
                    right_plane_ids.append(primitive_dict["planes"].shape[0])
                    primitive_dict["planes"] = np.vstack((primitive_dict["planes"], primitive_dict["planes"][id]))
                    primitive_dict["plane_ids"].append(primitive_dict["plane_ids"][id])
                    primitive_dict["halfspaces"].append(primitive_dict["halfspaces"][id])
                    primitive_dict["split_count"].append(primitive_dict["split_count"][id]+1)
                    primitive_dict["point_groups"].append(right_points)


                    # get all hull points and  make a new hull on the left side
                    if right_points.shape[0] > 2:
                        new_hull = ProjectedConvexHull(primitive_dict["convex_hulls"][id].plane_params, right_points)
                    else:
                        new_hull = primitive_dict["convex_hulls"][id]
                        new_hull.hull = None
                        new_hull.hull_points = right_points
                        new_hull.all_points = right_points
                    primitive_dict["convex_hulls"].append(new_hull)
                    hull_points = new_hull.hull_points

                    if hull_points.shape[0] <= self.vertex_group_n_fill:
                        fill = self.vertex_group_n_fill - hull_points.shape[0]
                        fill = hull_points[np.random.choice(hull_points.shape[0],fill)]
                        hull_points=np.concatenate((hull_points,fill))
                    else:
                        logger.warning("Fill value overflow. Len of hull points = {}, fill value = {}".format(hull_points.shape[0],self.vertex_group_n_fill))
                        hull_points = hull_points[:self.vertex_group_n_fill]
                    primitive_dict["hull_vertices"] = self.torch.cat(
                        (primitive_dict["hull_vertices"],
                         self.torch.Tensor(np.array(hull_points)).to('cuda')[None, :, :]))


                self.split_count+=1

                # planes[id, :] = np.nan
                # point_groups[id][:, :] = np.nan

        return left_plane_ids,right_plane_ids





    def _split_support_points(self,best_plane_id,current_ids,primitive_dict, th=1):

        '''
        :param best_plane_id:
        :param current_ids:
        :param planes:
        :param point_groups: padded 2d array of point groups with NaNs
        :param n_points_per_plane: real number of points per group (ie plane)
        :return: left and right planes
        '''

        best_plane = primitive_dict["planes"][current_ids[best_plane_id]]

        ### now put the planes into the left and right subspace of the best_plane split
        ### planes that lie in both subspaces are split (ie their point_groups are split) and appended as new planes to the planes array, and added to both subspaces
        left_plane_ids = []
        right_plane_ids = []
        for id in current_ids:

            if id == current_ids[best_plane_id]:
                continue

            # which_side = best_plane[0] * point_groups[id][:, 0] + best_plane[1] * point_groups[id][:, 1] + best_plane[2] * point_groups[id][:, 2] + best_plane[3]
            which_side = best_plane[0] * primitive_dict["point_groups"][id][:, 0] + best_plane[1] * primitive_dict["point_groups"][id][:, 1] + best_plane[2] * primitive_dict["point_groups"][id][:, 2]
            left_points = primitive_dict["point_groups"][id][which_side < -best_plane[3], :]
            right_points = primitive_dict["point_groups"][id][which_side > -best_plane[3], :]
            assert (primitive_dict["point_groups"][id].shape[0] > th)  # threshold cannot be bigger than the detection threshold

            if (primitive_dict["point_groups"][id].shape[0] - left_points.shape[0]) <= th:
                # if left_points.shape[0] > th:
                left_plane_ids.append(id)
                primitive_dict["point_groups"][id] = left_points  # update the point group, in case some points got dropped according to threshold
            elif(primitive_dict["point_groups"][id].shape[0] - right_points.shape[0]) <= th:
                # if right_points.shape[0] > th:
                right_plane_ids.append(id)
                primitive_dict["point_groups"][id] = right_points # update the point group, in case some points got dropped according to threshold
            else:
                # print("id:{}: total-left/right: {}-{}/{}".format(current_ids[best_plane_id],n_points_per_plane[id],left_points.shape[0],right_points.shape[0]))
                if (left_points.shape[0] > th):
                    left_plane_ids.append(primitive_dict["planes"].shape[0])
                    primitive_dict["point_groups"].append(left_points)
                    primitive_dict["planes"] = np.vstack((primitive_dict["planes"], primitive_dict["planes"][id]))
                    primitive_dict["plane_ids"].append(primitive_dict["plane_ids"][id])
                    primitive_dict["halfspaces"].append(primitive_dict["halfspaces"][id])
                    primitive_dict["split_count"].append(primitive_dict["split_count"][id]+1)
                if (right_points.shape[0] > th):
                    right_plane_ids.append(primitive_dict["planes"].shape[0])
                    primitive_dict["point_groups"].append(right_points)
                    primitive_dict["planes"] = np.vstack((primitive_dict["planes"], primitive_dict["planes"][id]))
                    primitive_dict["plane_ids"].append(primitive_dict["plane_ids"][id])
                    primitive_dict["halfspaces"].append(primitive_dict["halfspaces"][id])
                    primitive_dict["split_count"].append(primitive_dict["split_count"][id]+1)

                self.split_count+=1

                # planes[id, :] = np.nan
                # point_groups[id][:, :] = np.nan

        return left_plane_ids,right_plane_ids


    def _init_polygons(self):

        """
        3. initialize the polygons
        3a. intersects all pairs of polyhedra that share an edge in the graph and store the intersections on the edge
        3b. init an empty vertices list needed for self.construct_polygons
        """

        for e0,e1 in self.graph.edges:

            edge = self.graph.edges[e0,e1]
            c0 = self.cells.get(e0)
            c1 = self.cells.get(e1)
            ### this doesn't work, because after simplify some intersections are set, but are set with the wrong intersection from a collapse.
            ### I could just say if self.simplified or something like that, but for now will just recaculate all the intersections here
            # if not self.graph.edges[e0,e1]["intersection"]:
            intersection = c0.intersection(c1)
            if intersection.dim() == 2:
                edge["intersection"] = c0.intersection(c1)
                edge["vertices"] =  []
            else:
                self.graph.remove_edge(e0,e1)

            # edge["vertices"] = edge["intersection"].vertices_list()

        self.polygons_initialized = True



    def simplify(self):

        # TODO: try a simplify version from top to botton

        """
        2. simplify the partition
        :return:
        """

        ## this function does not need the sibling polygons to be initialized, ie edges need to be there, but we do not need to know the intersection!! initalizing afterwards is sufficient!



        def filter_edge(n0,n1):
            to_process = ((self.graph.nodes[n0]["occupancy"] == self.graph.nodes[n1]["occupancy"]) and self.graph.edges[n0,n1]["convex_intersection"])
            return to_process

        before=len(self.graph.nodes)
        edges = list(nx.subgraph_view(self.graph,filter_edge=filter_edge).edges)
        while len(edges):

            for c0,c1 in edges:
                nx.contracted_edge(self.graph, (c0, c1), self_loops=False, copy=False)

                parent = self.tree.parent(c0)
                pp_id = self.tree.parent(parent.identifier).identifier

                self.tree.remove_node(parent.identifier)

                dd = {"plane_ids": parent.data["plane_ids"]}
                # self.cells[c0] = self.cells[parent.identifier]
                self.cells[c0] = Polyhedron(vertices=self.cells[c0].vertices_list()+self.cells[c1].vertices_list())
                self.tree.create_node(tag=c0, identifier=c0, data=dd, parent=pp_id)

                if len(self.tree.siblings(c0)) == 0:
                    # TODO: maybe I can further simplify in this case by removing the alone sibling
                    continue

                sibling =  self.tree.siblings(c0)[0]
                if sibling.is_leaf():
                    self.graph.edges[c0, sibling.identifier]["convex_intersection"] = True

            edges = list(nx.subgraph_view(self.graph, filter_edge=filter_edge).edges)


        logger.info("Simplified partition from {} to {} cells".format(before,len(self.graph.nodes)))

        self.polygons_initialized = False


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

        logger.info("Add bounding planes...")

        pmin = vector(self.bounding_poly.bounding_box()[0])
        pmax = vector(self.bounding_poly.bounding_box()[1])
        d = pmax-pmin
        d = d*QQ(self.initial_padding*10)
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

        for i,plane in enumerate(bb_planes):

            self.plane_dict[str(plane)] = -(i+1)

            hspace_neg, hspace_pos = self._inequalities(plane)
            op = outside_poly.intersection(Polyhedron(ieqs=[hspace_neg]))

            if self.export:
                self.cellComplexExporter.write_cell(self.model,op,count=-(i+1))

            self.cells[-(i+1)] = op
            self.graph.add_node(-(i+1), occupancy=0.0)


            for cell_id in list(self.graph.nodes):

                intersection = op.intersection(self.cells.get(cell_id))

                if intersection.dim() == 2:
                    self.graph.add_edge(-(i+1),cell_id,intersection=None, vertices=[],
                                   supporting_plane=plane, convex_intersection=False, bounding_box_edge=True)


    def construct_polygons(self):

        """
        4. add missing vertices to the polyhedron facets by intersecting all neighbors with all neighbors
        :return:
        """

        if not self.polygons_initialized:
            self._init_polygons()

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
                    this_edge["vertices"]+=facet_intersection.vertices_list()

            for neighbor in list(self.graph[c1]):
                if neighbor == c0: continue
                this_edge = self.graph[c1][neighbor]
                facet_intersection = current_facet.intersection(this_edge["intersection"])
                if facet_intersection.dim() == 0 or facet_intersection.dim() == 1:
                    current_edge["vertices"] += facet_intersection.vertices_list()
                    this_edge["vertices"] += facet_intersection.vertices_list()


    def build_tree(self, model):

        # TODO: make a version where the tree is build without any intersection computations.
        #  Here the tree nodes are planes, and only the leaf nodes are (not yet) constructed cells (see Murali et al. Figure 3 for such a tree).
        # To construct the cells, (after the tree has been constructed) one simply has to walk up the tree from the leaf node to the root and collect the planes along it.
        # Planes are oriented depending on which side of the tree one is coming from e.g. each child could have a label positive or negative.
        # Stop walking once the collected planes form a bounded region -> the cell.
        # Now the cells can be labelled and simplified (siblings are known from the tree) and a convex decomposition can be extracted.
        # For surface reconstruction, i.e. to get the interface facets, a graph adjacency has to be recovered.
        # This could maybe be done by analysing from which supporting planes the cells come, i.e. if two cells share one supporting plane they are probably heighbors.

        pass


    @profile
    def construct_partition(self, m, mode=Tree.DEPTH, th=1, export=False, insertion_order="product-earlystop", device='cpu'):
        """
        1. construct the partition
        :param m:
        :param mode:
        :param th:
        :param ordering:                                                                                                                                                                                                                              
        :param export:
        :return:
        """
        logger.info('Construct partition with mode {} on {}'.format(insertion_order, self.device))
        if export:
            logger.warning('\nDebug export activated!\n')
        primitive_dict = dict()
        primitive_dict["planes"] = self.planes
        primitive_dict["halfspaces"] = list(self.halfspaces)
        primitive_dict["point_groups"] = list(self.points)
        primitive_dict["hull_vertices"] = self.hull_vertices
        primitive_dict["convex_hulls"] = list(self.convex_hulls)
        primitive_dict["split_count"] = [0]*len(self.planes)
        primitive_dict["plane_ids"] = list(range(self.planes.shape[0]))

        cell_count = 0
        self.split_count = 0

        ## init the graph
        self.graph = nx.Graph()
        self.graph.add_node(cell_count, convex=self.bounding_poly)

        ## expand the tree as long as there is at least one plane inside any of the subspaces
        self.tree = Tree()
        dd = {"plane_ids": np.arange(primitive_dict["planes"].shape[0])}
        self.tree.create_node(tag=cell_count, identifier=cell_count, data=dd)  # root node
        self.cells[cell_count] = self.bounding_poly
        children = self.tree.expand_tree(0, filter=lambda x: x.data["plane_ids"].shape[0], mode=mode)
        plane_count = 0
        n_points_total = np.concatenate(primitive_dict["point_groups"],dtype=object).shape[0]
        pbar = tqdm(total=n_points_total,file=sys.stdout)
        best_plane_ids = []
        for child in children:


            current_ids = self.tree[child].data["plane_ids"]
            current_cell = self.cells.get(child)
            plane_count+=1  # only used for debugging exports


            if len(current_ids) == 1:
                best_plane_id = 0
                best_plane = primitive_dict["planes"][current_ids[best_plane_id]]
                left_planes = []; right_planes = []
            else:
                best_plane_id = 0 if not insertion_order else self._get_best_plane(current_ids,primitive_dict, insertion_order)
                best_plane = primitive_dict["planes"][current_ids[best_plane_id]]
                ### split the primitives with the best_plane, and append them to the plane array
                if self.device == 'cpu':
                    left_planes, right_planes = self._split_support_points(best_plane_id,current_ids,primitive_dict, th)
                elif self.device == 'gpu':
                    left_planes, right_planes = self._split_support_points_gpu(best_plane_id,current_ids,primitive_dict, th)
                else:
                    raise NotImplementedError
            

            ### for debugging
            best_plane_ids.append(best_plane_id)

            # print("\n{}: {} with point {}".format(current_ids[best_plane_id],best_plane,primitive_dict["point_groups"][current_ids[best_plane_id]][0,:]))

            ### progress bar update
            n_points_processed = len(primitive_dict["point_groups"][current_ids[best_plane_id]])


            ### export best plane
            if export:
                epoints = primitive_dict["point_groups"][current_ids[best_plane_id]]
                epoints = epoints[~np.isnan(epoints).all(axis=-1)]
                if epoints.shape[0]>3:
                    color = self.plane_dict.get(str(best_plane))
                    color = self.plane_colors[color]
                    self.planeExporter.export_plane(os.path.dirname(m["planes"]), best_plane, epoints,count=str(plane_count),color=color)


            ## create the new convexes
            # hspace_positive, hspace_negative = primitive_dict["halfspaces"][current_ids[best_plane_id],0], primitive_dict["halfspaces"][current_ids[best_plane_id],1]
            hspace_positive, hspace_negative = primitive_dict["halfspaces"][current_ids[best_plane_id]][0], primitive_dict["halfspaces"][current_ids[best_plane_id]][1]

            cell_negative = current_cell.intersection(hspace_negative)
            cell_positive = current_cell.intersection(hspace_positive)

            ## update tree by creating the new nodes with the planes that fall into it
            ## and update graph with new nodes
            if(cell_negative.dim() == 3):
                if export:
                    self.cellComplexExporter.write_cell(m,cell_negative,count=str(cell_count+1)+"n")
                dd = {"plane_ids": np.array(left_planes)}
                cell_count = cell_count+1
                neg_cell_id = cell_count
                self.tree.create_node(tag=neg_cell_id, identifier=neg_cell_id, data=dd, parent=child)
                self.graph.add_node(neg_cell_id)
                self.cells[neg_cell_id] = cell_negative

            if(cell_positive.dim() == 3):
                if export:
                    self.cellComplexExporter.write_cell(m,cell_positive,count=str(cell_count+1)+"p")
                dd = {"plane_ids": np.array(right_planes)}
                cell_count = cell_count+1
                pos_cell_id = cell_count
                self.tree.create_node(tag=pos_cell_id, identifier=pos_cell_id, data=dd, parent=child)
                self.graph.add_node(pos_cell_id)
                self.cells[pos_cell_id] = cell_positive

            if(cell_positive.dim() == 3 and cell_negative.dim() == 3):
                self.graph.add_edge(neg_cell_id, pos_cell_id, intersection=None, vertices=[],
                               supporting_plane=best_plane, convex_intersection=True, bounding_box_edge=False)
                if export:
                    new_intersection = cell_negative.intersection(cell_positive)
                    self.cellComplexExporter.write_facet(m,new_intersection,count=plane_count)

            ## add edges to other cells, these must be neigbors of the parent (her named child) of the new subspaces
            neighbors_of_old_cell = list(self.graph[child])
            old_cell_id=child
            for neighbor_id_old_cell in neighbors_of_old_cell:
                logger.debug("make neighbors")

                # get the neighboring convex
                nconvex = self.cells.get(neighbor_id_old_cell)
                # intersect new cells with old neighbors to make the new facets
                n_nonempty = False; p_nonempty = False
                if cell_negative.dim()==3:
                    negative_intersection = nconvex.intersection(cell_negative)
                    n_nonempty = negative_intersection.dim()==2
                if cell_positive.dim()==3:
                    positive_intersection = nconvex.intersection(cell_positive)
                    p_nonempty = positive_intersection.dim()==2
                # add the new edges (from new cells with intersection of old neighbors) and move over the old additional vertices to the new
                if n_nonempty:
                    self.graph.add_edge(neighbor_id_old_cell,neg_cell_id,intersection=negative_intersection, vertices=[],
                                   supporting_plane=self.graph[neighbor_id_old_cell][old_cell_id]["supporting_plane"], convex_intersection=False, bounding_box_edge=False)
                if p_nonempty:
                    self.graph.add_edge(neighbor_id_old_cell, pos_cell_id, intersection=positive_intersection, vertices=[],
                                   supporting_plane=self.graph[neighbor_id_old_cell][old_cell_id]["supporting_plane"], convex_intersection=False, bounding_box_edge=False)

            self.graph.remove_node(child)
            pbar.update(n_points_processed)

            if self.device == 'gpu':
                primitive_dict["halfspaces"][current_ids[best_plane_id]] = None
                primitive_dict["point_groups"][current_ids[best_plane_id]] = None
                primitive_dict["convex_hulls"][current_ids[best_plane_id]] = None

            del self.cells[child]

        pbar.close()

        # print(best_plane_ids)
        # if export:
        #     self.cellComplexExporter.write_graph(m,self.graph,self.cells)
        self.polygons_initialized = False # false because I do not initialize the sibling facets

        logger.info("{} input planes were split {} times, making a total of {} planes now".format(len(self.planes),self.split_count,len(primitive_dict["planes"])))

        return 0


    def save_partition(self,infile):

        logger.info("Save tree, graph and convex cells to file...")

        os.makedirs(infile,exist_ok=True)

        if self.tree is not None:
            pickle.dump(self.tree,open(os.path.join(infile,'tree.pickle'),'wb'))
        pickle.dump(self.graph,open(os.path.join(infile,'graph.pickle'),'wb'))
        pickle.dump(self.cells,open(os.path.join(infile,'cells.pickle'),'wb'))



    def load_partition(self,infile):

        logger.info("Load tree, graph and convex cells from file...")

        if os.path.isfile(os.path.join(infile,'tree.pickle')):
            self.tree = pickle.load(open(os.path.join(infile,'tree.pickle'),'rb'))
        self.graph = pickle.load(open(os.path.join(infile,'graph.pickle'),'rb'))
        self.cells = pickle.load(open(os.path.join(infile,'cells.pickle'),'rb'))

        assert(len(self.cells) == len(self.graph.nodes)) ## this makes sure that every graph node has a convex attached

        self.polygons_initialized = False # false because I do not initialize the sibling facets


    @staticmethod
    def _pad_bound(bound, padding=0.00):
        """
        Pad bound.

        Parameters
        ----------
        bound: (2, 3) float
            Bound of the query planar primitive
        padding: float
            Padding factor, defaults to 0.05.

        Returns
        -------
        as_float: (2, 3) float
            Padded bound
        """

        extent = bound[1] - bound[0]
        return [bound[0] - extent * padding, bound[1] + extent * padding]


    def _intersect_bound_plane(self, bound, plane, exhaustive=False, epsilon=10e-5):
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
            intersection_bound = np.where(np.all(center_distance * 2 < extent_query + extent_targets + epsilon, axis=1))[0]

        # plane-AABB intersection test from extracted intersection_bound only
        # https://gdbooks.gitbooks.io/3dcollisions/content/Chapter2/static_aabb_plane.html
        # compute the projection interval radius of AABB onto L(t) = center + t * normal
        radius = np.dot(extent_targets[intersection_bound] / 2, np.abs(plane[:3]))
        # compute distance of box center from plane
        distance = np.dot(center_targets[intersection_bound], plane[:3]) + plane[3]
        # intersection between plane and AABB occurs when distance falls within [-radius, +radius] interval
        intersection_plane = np.where(np.abs(distance) <= radius + epsilon)[0]

        return intersection_bound[intersection_plane]

    @staticmethod
    def _inequalities(plane):
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

    def _index_node_to_cell(self, query):
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


    def _intersect_neighbour(self, kwargs):
        """
        Intersection test between partitioned cells and neighbouring cell.
        Implemented for multi-processing across all neighbours.

        Parameters
        ----------
        kwargs: (int, Polyhedron object, Polyhedron object, Polyhedron object)
            (neighbour index, positive cell, negative cell, neighbouring cell)
        """
        n, cell_positive, cell_negative, cell_neighbour = kwargs['n'], kwargs['positive'], kwargs['negative'], kwargs['neighbour']

        interface_positive = cell_positive.intersection(cell_neighbour)

        if interface_positive.dim() == 2:
            # this neighbour can connect with either or both children
            self.graph.add_edge(self.index_node + 1, n, supporting_plane=kwargs["supporting_plane"])
            interface_negative = cell_negative.intersection(cell_neighbour)
            if interface_negative.dim() == 2:
                self.graph.add_edge(self.index_node + 2, n, supporting_plane=kwargs["supporting_plane"])
        else:
            # this neighbour must otherwise connect with the other child
            self.graph.add_edge(self.index_node + 2, n, supporting_plane=kwargs["supporting_plane"])

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

        self.graph = nx.Graph()
        self.graph.add_node(0)  # the initial cell
        self.index_node = 0  # unique for every cell ever generated

        self.cells_bounds = [self.bounding_poly.bounding_box()]
        self.cells = [self.bounding_poly]
        cell_dict = dict()
        cell_dict[self.index_node] = self.bounding_poly

        if exhaustive:
            logger.info('construct exhaustive cell complex'.format())
        else:
            logger.info('construct cell complex'.format())

        tik = time.time()

        pool = None
        if num_workers > 0:
            pool = multiprocessing.Pool(processes=num_workers)

        pbar = trange(len(self.bounds))
        for i in pbar:  # kinetic for each primitive
            # bounding box intersection test
            # indices of existing cells with potential intersections
            indices_cells = self._intersect_bound_plane(self.bounds[i], self.planes[i], exhaustive)
            assert len(indices_cells), 'intersection failed! check the initial bound'

            # half-spaces defined by inequalities
            # no change_ring() here (instead, QQ() in _inequalities) speeds up 10x
            # init before the loop could possibly speed up a bit
            hspace_positive, hspace_negative = [Polyhedron(ieqs=[inequality]) for inequality in
                                                self._inequalities(self.planes[i])]


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
                    self.graph.add_edge(self.index_node + 1, self.index_node + 2,supporting_plane=self.planes[i])

                    # get neighbours of the current cell from the graph
                    neighbours = self.graph[list(self.graph.nodes)[index_cell]]  # index in the node list

                    if neighbours:
                        # get the neighbouring cells to the parent
                        cells_neighbours = [self.cells[self._index_node_to_cell(n)] for n in neighbours]

                        # adjacency test between both created cells and their neighbours
                        # todo:
                        #   Avoid 3d-3d intersection if possible. Unsliced neighbours connect with only one child;
                        #   sliced neighbors connect with both children.

                        kwargs = []
                        for n, cell in zip(neighbours, cells_neighbours):
                            supporting_plane = self.graph.edges[list(self.graph.nodes)[index_cell],n]["supporting_plane"]
                            kwargs.append({'n': n, 'positive': cell_positive, 'negative': cell_negative, 'neighbour': cell,
                                           'supporting_plane':supporting_plane})

                        if pool is None:
                            for k in kwargs:
                                self._intersect_neighbour(k)
                        else:
                            pool.map(self._intersect_neighbour, kwargs)

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

        logger.debug('cell complex constructed: {:.2f} s'.format(time.time() - tik))

        self.cells=cell_dict


