"""
complex.py
----------

Cell complex from planar primitive arrangement.

A linear cell complex is constructed from planar primitives
with adaptive binary space partitioning: upon insertion of a primitive
only the local cells that are intersecting it will be updated,
so will be the corresponding adjacency graph of the complex.
"""
import copy
import os
import time, multiprocessing, pickle, logging, trimesh
from pathlib import Path
from fractions import Fraction

import numpy as np
from tqdm import trange
import networkx as nx
from sage.all import QQ, RDF, ZZ, Polyhedron, vector, arctan2
from treelib import Tree
from tqdm import tqdm
import open3d as o3d
from collections import defaultdict
from copy import deepcopy
import gco # pip install gco-wrapper

from .export_complex import CellComplexExporter
from .imports import *
import libPyLabeler as PL
import libSoup2Mesh as s2m
from pyplane.export import PlaneExporter
from pyplane.pyplane import PyPlane, SagePlane, ProjectedConvexHull
from fancycolor.color import FancyColor

class CellComplex:
    """
    Class of cell complex from planar primitive arrangement.
    """
    def __init__(self, model, vertex_group, initial_padding=0.02, device='cpu', logger=None, debug_export=False):
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
        """
        self.model = model
        self.planeExporter = PlaneExporter()
        self.cellComplexExporter = CellComplexExporter(self)

        self.vg = vertex_group
        self.vg.input_planes = copy.deepcopy(self.vg.planes)

        self.logger = logger if logger else logging.getLogger("COMPOD")

        self.logger.debug('Init cell complex with padding {}'.format(initial_padding))

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
        self.logger.debug("Bounding planes will be appended to self.vg.input_planes")

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

        dtype = self.vg.input_planes.dtype
        self.vg.input_planes = np.vstack((self.vg.input_planes,np.flip(bb_planes,axis=0).astype(dtype)))

        max_node_id = max(list(self.graph.nodes))
        for i,plane in enumerate(bb_planes):


            hspace_neg, hspace_pos = self._inequalities(plane)
            op = outside_poly.intersection(Polyhedron(ieqs=[hspace_neg]))

            if self.debug_export:
                self.cellComplexExporter.write_cell(self.model,op,count=-(i+1))

            # self.cells[-(i+1)] = op
            # self.graph.add_node(-(i+1), occupancy=0.0)
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

    def _orient_polygon_exact(self, points, outside):
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


    def save_colored_soup(self, filename):

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

                # TODO: a better solution instead of using a plane dict is simply to get the ID from the self.vg.planes_ids array
                plane_id = self.graph.edges[e0, e1]["supporting_plane_id"]
                col = self.vg.plane_colors[plane_id] if plane_id > -1 else np.array([50,50,50])
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
                # for cgal export
                faces.append(np.arange(len(intersection_points)) + n_points)
                n_points += len(intersection_points)

        all_points = np.array(all_points, dtype=float)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.logger.debug('Save colored polygon soup to {}'.format(filename))

        self.cellComplexExporter.write_colored_soup_to_ply(filename, points=all_points, facets=faces, pcolors=pcolors, fcolors=fcolors)



    def save_simplified_surface(self, filename, backend = "python", triangulate = False):

        """
        This code works, but there is a bug in trimesh.Trimesh.outline(). See GitHub issue: https://github.com/mikedh/trimesh/issues/1934
        Result is that some simplified meshes are not watertight.

        :param filename:
        :param backend:
        :param triangulate:
        :return:
        """

        def _triangulate_points(v, attribute):
            n = len(v)
            triangles = []
            attributes = []
            for i in range(n - 2):
                tri = [0, i % n + 1, i % n + 2]
                triangles.append(v[tri])
                attributes.append(attribute)

            return triangles, attributes

        self.logger.info('Save simplified surface mesh ({})...'.format(backend))

        all_points = []
        all_triangles = []
        all_triangle_plane_ids = []
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
                ## here we want facets coming from the same plane to be oriented the same

                outside = vector(intersection_points[0])+vector([QQ(plane[1]),QQ(plane[2]),QQ(plane[3])])
                if self._orient_polygon_exact(intersection_points,outside):
                    intersection_points = np.flip(intersection_points, axis=0)

                all_points.append(intersection_points)
                
                polys = np.arange(len(intersection_points))+n_points
                
                tris, plane_ids = _triangulate_points(polys,plane_id)
                all_triangles.append(tris)
                all_triangle_plane_ids.append(plane_ids)


                n_points+=len(intersection_points)


        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.logger.debug('Save polygon with backend {} mesh to {}'.format(backend,filename))
        
        ## make a mesh, where each face of the partition is a triangulated face of the mesh
        all_points = np.concatenate(all_points).astype(float)
        all_triangles = np.concatenate(all_triangles)
        all_triangle_plane_ids = np.concatenate(all_triangle_plane_ids)
        atpd = {}
        for i,id in enumerate(all_triangle_plane_ids):
            atpd[i] = id
        mesh = trimesh.Trimesh(vertices=all_points,faces=all_triangles,face_attributes=atpd)
        mesh.merge_vertices()
        emesh = deepcopy(mesh)
        emesh.face_attributes = dict()
        emesh.export(filename[:-4]+"_trimesh.ply")


        ## collect the regions of the mesh, ie facets that came from the same plane
        region_dict = defaultdict(list)
        for i,f in enumerate(mesh.face_attributes.values()):
            region_dict[f]+=[i]
            
        polygons = []
        polygon_lens = []

        from triangle import triangulate

        ## apply a constrained delaunay triangulation to the regions
        for plane_id,faces in region_dict.items():

            ## get outlines of the region
            outline3d = mesh.outline(face_ids=faces)
            pyplane = PyPlane(self.planes[plane_id])
            verts_2d = pyplane.to_2d(mesh.vertices)
            outline2d = trimesh.path.Path2D(entities=deepcopy(outline3d.entities),
                            vertices=deepcopy(verts_2d),
                            process=False)

            referenced_vertices = deepcopy(outline2d.referenced_vertices)
            outline2d.remove_unreferenced_vertices()

            ## collect holes per outline
            points_inside_holes = []
            for exterior_id, interior_ids in outline2d.enclosure_shell.items():

                # this is simply for finding a point inside the hole, so I can pass it to the constrained delaunay triangulator
                for iid in interior_ids:
                    ent = outline2d.entities[iid]
                    hpts = outline2d.vertices[ent.points, :]
                    tridict = {"vertices": hpts}
                    if not ent.closed: # here is the bug mentioned in the header of this function. All outlines should be closed, but self-intersecting ones are not.
                        self.logger.warning("Skipping wholes")
                        continue
                    hole = triangulate(tridict)
                    pts = hole['vertices'][hole["triangles"][0]]
                    points_inside_holes.append(pts.mean(axis=0))

            points_inside_holes = np.array(points_inside_holes)

            if len(points_inside_holes):
                tridict = {"vertices": outline2d.vertices, "segments": outline2d.vertex_nodes, "holes": points_inside_holes}
            else:
                tridict = {"vertices": outline2d.vertices, "segments": outline2d.vertex_nodes}

            tri = triangulate(tridict, 'p') # 'p' means make a Delaunay triangulation constraint to the passed segments
            points2d = tri['vertices']
            triangles = tri['triangles']
            triangles = referenced_vertices[triangles]

            polygons.append(triangles)
            polygon_lens.append(np.zeros(triangles.shape[0],dtype=int)+3)

        verts=np.array(mesh.vertices)

        if backend == "cgal":
            sm = s2m.Soup2Mesh()
            sm.loadSoup(verts, np.concatenate(polygon_lens, dtype=int), np.concatenate(polygons, dtype=int).flatten())
            sm.makeMesh(True)
            sm.saveMesh(filename)
        elif backend == "python":
            mesh = trimesh.Trimesh(vertices=verts,faces=np.concatenate(polygons))
            mesh.export(filename)
        else:
            raise NotImplementedError


    def save_surface(self, filename, backend = "python", triangulate = False):

        self.logger.info('Save surface mesh ({})...'.format(backend))


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

                plane_id = self.graph[e0][e1]["supporting_plane_id"]
                plane = self.vg.input_planes[plane_id]
                correct_order = self._sort_vertex_indices_by_angle_exact(intersection_points,plane)

                assert(len(intersection_points)==len(correct_order))
                intersection_points = intersection_points[correct_order]

                if(len(intersection_points)<3):
                    print("ERROR: Encountered facet with less than 3 vertices.")
                    sys.exit(1)

                ## orient polygon
                outside = self.cells.get(e0).center() if c1["occupancy"] else self.cells.get(e1).center()
                if self._orient_polygon_exact(intersection_points,outside):
                    intersection_points = np.flip(intersection_points, axis=0)

                for i in range(intersection_points.shape[0]):
                    all_points.append(tuple(intersection_points[i,:]))
                tris.append(intersection_points)
                # for cgal export
                faces.append(np.arange(len(intersection_points))+n_points)
                face_lens.append(len(intersection_points))
                n_points+=len(intersection_points)

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        if backend == "cgal":
            sm = s2m.Soup2Mesh()
            sm.loadSoup(np.array(all_points,dtype=float), np.array(face_lens,dtype=int), np.concatenate(faces,dtype=int))
            sm.makeMesh(triangulate)
            sm.saveMesh(filename)
        elif backend == "python":
            if triangulate:
                self.logger.warning("Mesh will not be triangulated. Choose backend 'cgal' or 'trimesh' instead.")
            pset = set(all_points)
            pset = np.array(list(pset),dtype=object)
            facets = []
            for tri in tris:
                face = []
                for pt in tri:
                    face.append(np.argwhere((np.equal(pset,pt,dtype=object)).all(-1))[0][0])
                facets.append(face)
            self.cellComplexExporter.write_surface_to_off(filename,points=np.array(pset,dtype=np.float32),facets=facets)
        elif backend == "trimesh":
            if not triangulate:
                self.logger.warning("Mesh will be triangulated. Choose backend 'python' or 'cgal' instead.")
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
            mesh = trimesh.Trimesh(vertices=all_points,faces=np.concatenate(tris))
            mesh.export(filename)
        else:
            raise NotImplementedError


    def save_in_cells_explode(self,filename,shrink_percentage=0.01):

        self.logger.info('Save exploded inside cells...')

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


    def save_in_cells(self,filename):

        self.logger.info('Save inside cells...')

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
                primitive_ids.append([self.vg.plane_order[plane_id]])
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


    def _graph_cut(self,occs,binary_weight=0.2):

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

        # data_cost = prediction
        data_cost = np.array(occs, dtype=dtype)*4

        ### append high cost for inside for infinite cell
        # data_cost = np.append(data_cost, np.array([[-10000, 10000]]), axis=0)
        gc.set_data_cost(data_cost)
        smooth = (1 - np.eye(2)).astype(dtype)
        gc.set_smooth_cost(smooth)

        edges = np.array(graph.edges)

        edge_weight = np.ones(edges.shape[0], dtype=dtype)*4/edges.shape[0]

        gc.set_all_neighbors(edges[:, 0], edges[:, 1], edge_weight * binary_weight)

        for i, l in enumerate(labels):
            gc.init_label_at_site(i, l)

        self.logger.debug("Energy before GC (D + λ*S): {:.2g} + {:.2g}*{:.2g} = {:.2g}".format(gc.compute_data_energy(), binary_weight, gc.compute_smooth_energy()/binary_weight,
                                                            gc.compute_data_energy()+gc.compute_smooth_energy()))

        gc.expansion()
        self.logger.debug("Energy after GC (D + λ*S): {:.2g} + {:.2g}*{:.2g} = {:.2g}".format(gc.compute_data_energy(), binary_weight, gc.compute_smooth_energy()/binary_weight,
                                                            gc.compute_data_energy() + gc.compute_smooth_energy()))

        labels = gc.get_labels()

        return labels

    def label_partition(self,outpath=None,**args):

        self.logger.info('Label {} cells...'.format(len(self.graph.nodes)))

        if args["type"] == "pc":
            occs = self.label_partition_with_point_normals()
        elif args["type"] == "mesh":
            occs = self.label_partition_with_mesh(args["n_test_points"])
        else:
            self.logger.error("{} is not a valid labelling type. Choose either 'pc' or 'mesh'.".format(args["type"]))
            raise NotImplementedError


        if outpath is not None:
            # save occupancies to file
            np.savez(os.path.join(outpath,"occupancies.npz"),occupancies=occs,type=args["type"])

        # foccs = dict(zip(self.graph.nodes, occs))
        # nx.set_node_attributes(self.graph,occs,"float_occupancy")
        #
        if args["graph_cut"]:
            # occs = self.graph_cut(occs)
            occs = self._graph_cut(occs,args["binary_weight"])
        else:
            occs = occs[:,0]>occs[:,1]
        
        assert len(occs) == len(self.graph.nodes), "Number of cccupancy labels and graph nodes is not the same."
        occs = dict(zip(self.graph.nodes, np.rint(occs).astype(int)))

        nx.set_node_attributes(self.graph,occs,"occupancy")


    def label_partition_with_mesh(self, n_test_points=50):
        """
        Compute occupancy of each cell in the partition using a ground truth mesh and point sampling.
        :param m:
        :param n_test_points:
        :param graph_cut:
        :param export:
        :return:
        """

        pl=PL.PyLabeler(n_test_points)
        if pl.loadMesh(self.model["mesh"]):
            return 1
        points = []
        points_len = []
        # for i,id in enumerate(list(self.graph.nodes)):
        for id,cell in self.cells.items():
            # if id < 0:  continue # skip the bounding box cells
            bb = self.graph.nodes[id].get("bounding_box",0)
            if bb:  continue # skip the bounding box cells
            # cell = self.cells.get(id)
            if self.debug_export:
                self.cellComplexExporter.write_cell(self.model,cell,count=id,subfolder="final_cells")
            pts = np.array(cell.vertices())
            points.append(pts)
            # print(pts)
            points_len.append(pts.shape[0])


        # assert(isinstance(points[0].dtype,np.float32))
        occs = pl.labelCells(np.array(points_len),np.concatenate(points,axis=0))
        del pl

        occs = np.hstack((occs,np.zeros(6))) # this is for the bounding box cells, which are the last 6 cells of the graph
        occs = np.array([occs,1-occs]).transpose()

        return occs



    def _collect_facet_points(self):
        # def convex_contains1(convex, points):
        #     """
        #     If point is left of all support lines, it is contained by the convex
        #     :param convex:
        #     :param points:
        #     :return:
        #     """
        #     ineqs = np.array(convex.inequalities())
        #     c = ineqs[:, 0, np.newaxis]
        #     a = ineqs[:, 1, np.newaxis]
        #     b = ineqs[:, 2, np.newaxis]
        #     x = points[np.newaxis, :, 0]
        #     y = points[np.newaxis, :, 1]
        #     k = (a * x + b * y) >= -c
        #     return k.all(axis=0)
        def convex_contains2(vertices, points):
            """
            Check if points are contained in a convex polygon.
            :param vertices: The vertices of the convex polygon.
            :param points: The points array.
            :return:
            """
            sides = []
            n = len(vertices)
            for i in range(n):
                x0 = vertices[i, 0]
                x1 = vertices[(i + 1) % n, 0]
                y0 = vertices[i, 1]
                y1 = vertices[(i + 1) % n, 1]
                sides.append((points[:, 1] - y0) * (x1 - x0) - (points[:, 0] - x0) * (y1 - y0))

            sides = np.array(sides)
            sides1 = sides < 0
            sides2 = ~sides1 # orientation of the facet is unknown, so need to check for both sides
            sides1 = sides1.all(axis=0)
            sides2 = sides2.all(axis=0)
            sides = np.logical_or(sides1,sides2)
            return sides



        point_ids_dict = dict()
        count=0
        for e0, e1 in self.graph.edges:

            edge = self.graph.edges[e0, e1]
            if edge["supporting_plane_id"] < 0:  # bounding box planes have no associated points, so skip them
                point_ids_dict[(e0, e1)] = np.empty(shape=0,dtype=np.int32)
                continue

            group = self.vg.groups[edge["group_id"]]

            ## use my own contains functions because no need to do this with QQ bqse_ring and changing it to RDF is not robust.
            ## this works but is slow
            # polygon = edge["intersection"].affine_hull_projection()
            # contain = convex_contains1(polygon, self.vg.projected_points[group])
            ## this is much faster and gives the same result
            vertices = np.array(edge["intersection"].vertices())
            vertices = vertices[self._sorted_vertex_indices(edge["intersection"].adjacency_matrix())]
            contain = convex_contains2(vertices, self.vg.projected_points[group])

            point_ids_dict[(e0, e1)] = group[contain]

            if self.debug_export:
                pts = self.vg.points[group[contain]]
                count += 1
                col = np.random.randint(0, 255, size=3)
                if len(pts):
                    self.cellComplexExporter.write_points(self.model, pts, subfolder="labelling_facets", count=str(count) + "c", color=col)
                    # self.cellComplexExporter.write_facet(self.model, edge["intersection"], subfolder="labelling_facets", count=count, color=col)

        nx.set_edge_attributes(self.graph,point_ids_dict,"point_ids")

    def _collect_node_votes(self):
        # occupancy_dict = dict()
        occs = []
        for node in self.graph.nodes:

            if self.graph.nodes[node].get("bounding_box",0):
                # occupancy_dict[node] = (0,1000)
                occs.append([0,1000])
                continue

            centroid = np.array(self.cells[node].vertices()).mean(axis=0)
            inside_weight = 0; outside_weight = 0
            cell_points = []
            cell_normals = []
            for edge in self.graph.edges(node):
                edge = self.graph.edges[edge]
                pts = self.vg.points[edge["point_ids"]]
                if not len(pts):
                    continue
                inside_vectors = centroid - pts
                normal_vectors = self.vg.normals[edge["point_ids"]]
                if self.debug_export:
                    cell_points.append(pts)
                    cell_normals.append(normal_vectors)
                dp=(normal_vectors * inside_vectors).sum(axis=1)
                inside_weight+=(dp<0).sum()
                outside_weight+=(dp>0).sum()

            if self.debug_export:
                if len(cell_points):
                    st = "in" if inside_weight <= outside_weight else "out"
                    color = np.random.randint(0, 255, size=3)
                    self.cellComplexExporter.write_cell(self.model,self.cells[node],
                                                        count=str(node)+st,subfolder="labelling_cells",color=color)
                    self.cellComplexExporter.write_points(self.model, color=color, count=str(node)+st,
                                                          points=np.concatenate(cell_points),normals=np.concatenate(cell_normals),
                                                          subfolder="labelling_cells")

            # occupancy_dict[node] = (inside_weight,outside_weight)
            occs.append([inside_weight,outside_weight])

        return np.array(occs)/(2*len(self.vg.points))
        # nx.set_node_attributes(self.graph,occupancy_dict,"occupancy")


    def label_partition_with_point_normals(self):
        """
        Compute the occupancy of each cell of the partition according to the normal criterion introduced in Kinetic Shape Reconstruction [Bauchet & Lafarge 2020] (Section 4.2).
        :param outpath:
        :param graph_cut:
        :return:
        """

        self._collect_facet_points()
        return self._collect_node_votes()



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

        planes = self.vg.planes[current_ids]
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


    def _get_best_split_gpu(self,current_ids,primitive_dict,insertion_order="product-earlystop"):


        earlystop = False
        if "earlystop" in insertion_order:
            earlystop = True

        planes = self.vg.planes[current_ids]
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
            for i,plane in enumerate(self.vg.planes[current_ids]):
                if pgs[i].shape[0] > 2:
                    mesh = PyPlane(plane).get_trimesh_of_projected_points(pgs[i])
                    areas.append(mesh.area)
                else:
                    areas.append(0)
            return np.argmax(areas)
        else:
            raise NotImplementedError


    def _split_support_points(self,best_plane_id,current_ids,th=1):
        '''
        :param best_plane_id:
        :param current_ids:
        :param planes:
        :param point_groups: padded 2d array of point groups with NaNs
        :param n_points_per_plane: real number of points per group (ie plane)
        :return: left and right planes
        '''

        assert th >= 0,"Threshold must be >= 0"

        best_plane = self.vg.planes[current_ids[best_plane_id]]


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

            if (this_group.shape[0] - left_point_ids.shape[0]) <= th:
                left_plane_ids.append(id)
            elif(this_group.shape[0] - right_point_ids.shape[0]) <= th:
                right_plane_ids.append(id)
            else:
                if (left_point_ids.shape[0] > th):
                    left_plane_ids.append(self.vg.planes.shape[0])
                    self.vg.planes = np.vstack((self.vg.planes, self.vg.planes[id]))
                    self.vg.planes_ids.append(self.vg.planes_ids[id])
                    self.vg.halfspaces.append(self.vg.halfspaces[id])
                    self.vg.groups.append(left_point_ids)

                    # if not enough points for making a convex hull we simply keep the points 
                    
                    # get all hull points and  make a new hull on the left side
                    if left_point_ids.shape[0] > 2:
                        new_hull = ProjectedConvexHull(self.vg.planes[id],self.vg.points[left_point_ids])
                        new_group = left_point_ids[new_hull.hull.vertices]
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
                            "Fill value overflow. Len of hull points = {}, fill value = {}. Increase vg.n_fill in primitive.py for more robustness.".format(new_group.shape[0],self.vg.n_fill))
                        new_group = new_group[:self.vg.n_fill]

                    self.vg.hull_vertices = np.vstack((self.vg.hull_vertices,new_group))

                if (right_point_ids.shape[0] > th):
                    right_plane_ids.append(self.vg.planes.shape[0])
                    self.vg.planes = np.vstack((self.vg.planes, self.vg.planes[id]))
                    self.vg.planes_ids.append(self.vg.planes_ids[id])
                    self.vg.halfspaces.append(self.vg.halfspaces[id])
                    self.vg.groups.append(right_point_ids)

                    # if not enough points for making a convex hull we simply keep the points

                    # get all hull points and  make a new hull on the right side
                    if right_point_ids.shape[0] > 2:
                        new_hull = ProjectedConvexHull(self.vg.planes[id], self.vg.points[right_point_ids])
                        new_group = right_point_ids[new_hull.hull.vertices]
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
                            "Fill value overflow. Len of hull points = {}, fill value = {}. Increase vg.n_fill in primitive.py for more robustness.".format(new_group.shape[0],self.vg.n_fill))
                        new_group = new_group[:self.vg.n_fill]

                    self.vg.hull_vertices = np.vstack((self.vg.hull_vertices, new_group))

                self.split_count+=1

                # planes[id, :] = np.nan
                # point_groups[id][:, :] = np.nan

        return left_plane_ids,right_plane_ids


    def simplify_partition_graph_based(self):

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
        4. add missing vertices to the polyhedron facets by intersecting all neighbors with all neighbors
        :return:
        """

        if not self.polygons_initialized:
            self._init_polygons()

        self.logger.info("Construct polygons...")

        polygon_dict = defaultdict(trimesh.Trimesh)

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
                    # this_edge["vertices"]+=facet_intersection.vertices_list()

            for neighbor in list(self.graph[c1]):
                if neighbor == c0: continue
                this_edge = self.graph[c1][neighbor]
                facet_intersection = current_facet.intersection(this_edge["intersection"])
                if facet_intersection.dim() == 0 or facet_intersection.dim() == 1:
                    current_edge["vertices"] += facet_intersection.vertices_list()
                    # this_edge["vertices"] += facet_intersection.vertices_list()


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




    def construct_partition(self, mode=Tree.DEPTH, th=0, insertion_order="product-earlystop", device='cpu'):
        """
        1. construct the partition
        :param m:
        :param mode:
        :param th:
        :param ordering:                                                                                                                                                                                                                              
        :param export:
        :return:
        """
        self.logger.info('Construct partition with mode {} on {}'.format(insertion_order, self.device))

        self.vg.planes_ids = list(range(self.vg.planes.shape[0]))

        cell_count = 0
        self.split_count = 0

        ## init the graph
        self.graph = nx.Graph()
        self.graph.add_node(cell_count, convex=self.bounding_poly)

        ## expand the tree as long as there is at least one plane inside any of the subspaces
        self.tree = Tree()
        dd = {"plane_ids": np.arange(self.vg.planes.shape[0])}
        self.tree.create_node(tag=cell_count, identifier=cell_count, data=dd)  # root node
        self.cells[cell_count] = self.bounding_poly
        children = self.tree.expand_tree(0, filter=lambda x: x.data["plane_ids"].shape[0], mode=mode)
        plane_count = 0 # only used for debugging exports
        pbar = tqdm(total=np.concatenate(self.vg.groups).shape[0],file=sys.stdout)
        best_plane_ids = [] # only used for debugging exports
        for child in children:

            current_ids = self.tree[child].data["plane_ids"]
            current_cell = self.cells.get(child)

            if len(current_ids) == 1:
                best_plane_id = 0
                best_plane = self.vg.planes[current_ids[best_plane_id]]
                left_planes = []; right_planes = []
            else:
                best_plane_id = 0 if not insertion_order else self._get_best_plane(current_ids, insertion_order)
                best_plane = self.vg.planes[current_ids[best_plane_id]]
                ### split the point sets with the best_plane, and append the split sets to the self.vg arrays
                left_planes, right_planes = self._split_support_points(best_plane_id, current_ids, th)


            ### for debugging
            best_plane_id_input = self.vg.planes_ids[current_ids[best_plane_id]]
            best_plane_ids.append(best_plane_id_input)

            ### progress bar update
            n_points_processed = len(self.vg.groups[current_ids[best_plane_id]])

            ### export best plane
            if self.debug_export:
                plane_count += 1
                epoints = self.vg.groups[current_ids[best_plane_id]]
                epoints = self.vg.points[epoints]
                # epoints = epoints[~np.isnan(epoints.astype(float)).all(axis=-1)]
                if epoints.shape[0]>3:
                    color = self.vg.plane_colors[best_plane_id_input]
                    self.planeExporter.save_plane(os.path.dirname(self.model["planes"]), best_plane, epoints,count=str(plane_count),color=color)


            ## create the new convexes
            # hspace_positive, hspace_negative = self.vg.halfspaces[current_ids[best_plane_id],0], self.vg.halfspaces[current_ids[best_plane_id],1]
            hspace_positive, hspace_negative = self.vg.halfspaces[current_ids[best_plane_id]][0], self.vg.halfspaces[current_ids[best_plane_id]][1]

            cell_negative = current_cell.intersection(hspace_negative)
            cell_positive = current_cell.intersection(hspace_positive)

            ## update tree by creating the new nodes with the planes that fall into it
            ## and update graph with new nodes
            if(cell_negative.dim() == 3):
                if self.debug_export:
                    self.cellComplexExporter.write_cell(self.model,cell_negative,count=str(cell_count+1)+"n")
                dd = {"plane_ids": np.array(left_planes)}
                cell_count = cell_count+1
                neg_cell_id = cell_count
                self.tree.create_node(tag=neg_cell_id, identifier=neg_cell_id, data=dd, parent=child)
                self.graph.add_node(neg_cell_id)
                self.cells[neg_cell_id] = cell_negative

            if(cell_positive.dim() == 3):
                if self.debug_export:
                    self.cellComplexExporter.write_cell(self.model,cell_positive,count=str(cell_count+1)+"p")
                dd = {"plane_ids": np.array(right_planes)}
                cell_count = cell_count+1
                pos_cell_id = cell_count
                self.tree.create_node(tag=pos_cell_id, identifier=pos_cell_id, data=dd, parent=child)
                self.graph.add_node(pos_cell_id)
                self.cells[pos_cell_id] = cell_positive

            ## add the split plane to the parent node of the tree
            self.tree.nodes[child].data["supporting_plane_id"] = best_plane_id_input

            if(cell_positive.dim() == 3 and cell_negative.dim() == 3):

                new_intersection = cell_negative.intersection(cell_positive)
                self.graph.add_edge(neg_cell_id, pos_cell_id, intersection = new_intersection, vertices = [], group_id = current_ids[best_plane_id],
                               supporting_plane_id = best_plane_id_input, bounding_box_edge = False)
                if self.debug_export:
                    self.cellComplexExporter.write_facet(self.model,new_intersection,count=plane_count)

            ## add edges to other cells, these must be neigbors of the parent (her named child) of the new subspaces
            neighbors_of_old_cell = list(self.graph[child])
            old_cell_id=child
            for neighbor_id_old_cell in neighbors_of_old_cell:
                # self.logger.debug("make neighbors")

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
                    self.graph.add_edge(neighbor_id_old_cell,neg_cell_id,intersection=negative_intersection, vertices=[], group_id=self.graph[neighbor_id_old_cell][old_cell_id]["group_id"],
                                   supporting_plane_id=self.graph[neighbor_id_old_cell][old_cell_id]["supporting_plane_id"], convex_intersection=False, bounding_box_edge=False)
                if p_nonempty:
                    self.graph.add_edge(neighbor_id_old_cell, pos_cell_id, intersection=positive_intersection, vertices=[], group_id=self.graph[neighbor_id_old_cell][old_cell_id]["group_id"],
                                   supporting_plane_id=self.graph[neighbor_id_old_cell][old_cell_id]["supporting_plane_id"], convex_intersection=False, bounding_box_edge=False)

            self.graph.remove_node(child)
            pbar.update(n_points_processed)

            # if self.device == 'gpu':
            #     self.vg.halfspaces[current_ids[best_plane_id]] = None
            #     primitive_dict["point_groups"][current_ids[best_plane_id]] = None
            #     self.vg.convex_hulls[current_ids[best_plane_id]] = None

            del self.cells[child]

        pbar.close()

        self.polygons_initialized = False # false because I do not initialize the sibling facets

        self.logger.debug("Plane insertion order {}".format(best_plane_ids))
        self.logger.debug("{} input planes were split {} times, making a total of {} planes now".format(len(self.vg.input_planes),self.split_count,len(self.vg.planes)))

        return 0


    def save_partition_to_pickle(self,outpath):

        self.logger.info("Save tree, graph and convex cells to file...")

        os.makedirs(outpath,exist_ok=True)

        if self.tree is not None:
            pickle.dump(self.tree,open(os.path.join(outpath,'tree.pickle'),'wb'))
        pickle.dump(self.graph,open(os.path.join(outpath,'graph.pickle'),'wb'))
        pickle.dump(self.cells,open(os.path.join(outpath,'cells.pickle'),'wb'))
        pickle.dump(self.vg.groups,open(os.path.join(outpath,'groups.pickle'),'wb'))




    def load_partition_from_pickle(self,inpath):

        self.logger.info("Load tree, graph and convex cells from file...")

        self.tree = pickle.load(open(os.path.join(inpath,'tree.pickle'),'rb'))
        self.graph = pickle.load(open(os.path.join(inpath,'graph.pickle'),'rb'))
        self.cells = pickle.load(open(os.path.join(inpath,'cells.pickle'),'rb'))
        self.vg.groups = pickle.load(open(os.path.join(inpath,'groups.pickle'),'rb'))

        assert(len(self.cells) == len(self.graph.nodes)) ## this makes sure that every graph node has a convex attached

        self.polygons_initialized = False # false because I do not initialize the sibling facets

    def load_occupancies(self,inpath):

        occs = np.load(os.path.join(inpath,'occupancies.npz'))["occupancies"]
        occs = np.hstack((occs,[0,0,0,0,0,0]))
        occs = np.rint(occs)

        occs = dict(zip(self.graph.nodes, np.rint(occs).astype(int)))

        assert len(occs) == len(self.graph.nodes)

        nx.set_node_attributes(self.graph,occs,"occupancy")




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

        pbar = trange(len(self.bounds),file=sys.stdout)
        for i in pbar:  # kinetic for each primitive
            # bounding box intersection test
            # indices of existing cells with potential intersections
            indices_cells = _intersect_bound_plane(self.bounds[i], self.planes[i], exhaustive)
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
                    self.graph.add_edge(self.index_node + 1, self.index_node + 2,supporting_plane_id=i)

                    # get neighbours of the current cell from the graph
                    neighbours = self.graph[list(self.graph.nodes)[index_cell]]  # index in the node list

                    if neighbours:
                        # get the neighbouring cells to the parent
                        cells_neighbours = [self.cells[_index_node_to_cell(n)] for n in neighbours]

                        # adjacency test between both created cells and their neighbours
                        # todo:
                        #   Avoid 3d-3d intersection if possible. Unsliced neighbours connect with only one child;
                        #   sliced neighbors connect with both children.

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


