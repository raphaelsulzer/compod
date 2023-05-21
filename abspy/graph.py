"""
graph.py
----------

Adjacency graph of the cell complex.

Each cell in the complex is represented as a node in the graph.
In addition, two nodes (S and T) are appended in the graph,
representing foreground and background, respectively.
There are two kinds of edges in the graph: n-links and st-links.
An n-link exists in between of two adjacent cells.
An st-link connects every cell to S and to T.
"""
import os,sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
from sage.all import RR

from scipy.spatial import ConvexHull

from .logger import attach_to_log

import trimesh

logger = attach_to_log()

sys.path.append("/home/rsulzer/cpp/compact_mesh_reconstruction/build/release/Benchmark/Soup2Mesh")
import libSoup2Mesh as s2m

class AdjacencyGraph:
    """
    Class Adjacency graph of the cell complex.
    """

    def __init__(self, graph=None, quiet=False):
        """
        Init AdjacencyGraph.

        Parameters
        ----------
        graph: None or networkx Graph
            Graph object
        quiet: bool
            Disable logging if set True
        """
        if quiet:
            logger.disabled = True

        self.graph = graph
        self.uid = list(graph.nodes) if graph else None  # passed graph.nodes are sorted
        self.reachable = None  # for outer surface extraction
        self.non_reachable = None
        self._cached_interfaces = {}

    def load_graph(self, filepath):
        """
        Load graph from an external file.

        Parameters
        ----------
        filepath: str or Path
            Filepath to networkx graph
        """
        filepath = Path(filepath)
        if filepath.suffix == '.adjlist':
            logger.info('loading graph from {}'.format(filepath))
            self.graph = nx.read_adjlist(filepath)
            self.uid = self._sort_uid()  # loaded graph.nodes are unordered string
        else:
            raise NotImplementedError('file format not supported: {}'.format(filepath.suffix))

    def assign_weights_to_n_links(self, cells, attribute='area_overlap', normalise=True, factor=1.0, engine='Qhull',
                                  cache_interfaces=False):
        """
        Assign weights to edges between every cell.

        Parameters
        ----------
        cells: list of Polyhedra objects
            Polyhedra cells
        attribute: str
            Attribute to use for encoding n-links, options are 'radius_overlap',
            'area_overlap', 'vertices_overlap', 'area_misalign' and 'volume_difference'
        normalise: bool
            Normalise the attribute if set True
        factor: float
            Factor to multiply to the attribute
        engine: str
            Engine to compute convex hull
            'Qhull' is supported at the moment
        cache_interfaces: bool
            Cache interfaces if set True
        """

        radius = [None] * len(self.graph.edges)
        area = [None] * len(self.graph.edges)
        volume = [None] * len(self.graph.edges)
        num_vertices = [None] * len(self.graph.edges)

        if attribute == 'radius_overlap':
            for i, (m, n) in enumerate(self.graph.edges):
                # compute interface
                interface = cells[self._uid_to_index(m)].intersection(cells[self._uid_to_index(n)])
                if cache_interfaces:
                    self._cached_interfaces[m, n] = interface  # uid pair as key
                radius[i] = RR(interface.radius())

            for i, (m, n) in enumerate(self.graph.edges):
                max_radius = max(radius)
                # small (sum of) overlapping radius -> large capacity -> small cost -> cut here
                self.graph[m][n].update({'capacity': ((max_radius - radius[
                    i]) / max_radius if normalise else max_radius - radius[i]) * factor})

        elif attribute == 'area_overlap':
            for i, (m, n) in enumerate(self.graph.edges):
                # compute interface
                interface = cells[self._uid_to_index(m)].intersection(cells[self._uid_to_index(n)])
                if cache_interfaces:
                    self._cached_interfaces[m, n] = interface  # uid pair as key
                if engine == 'Qhull':
                    # 'volume' is the area of the convex hull when input points are 2-dimensional
                    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html
                    try:
                        area[i] = ConvexHull(interface.affine_hull_projection().vertices_list()).volume
                    except:
                        # degenerate floating-point
                        area[i] = RR(interface.affine_hull_projection().volume())
                else:
                    # slower computation
                    area[i] = RR(interface.affine_hull_projection().volume())

            for i, (m, n) in enumerate(self.graph.edges):
                max_area = max(area)
                # balloon term
                # small (sum of) overlapping area -> large capacity -> small cost -> cut here
                self.graph[m][n].update(
                    {'capacity': ((max_area - area[i]) / max_area if normalise else max_area - area[i]) * factor})

        elif attribute == 'vertices_overlap':
            # number of vertices on overlapping areas
            for i, (m, n) in enumerate(self.graph.edges):
                # compute interface
                interface = cells[self._uid_to_index(m)].intersection(cells[self._uid_to_index(n)])
                if cache_interfaces:
                    self._cached_interfaces[m, n] = interface  # uid pair as key
                num_vertices[i] = interface.n_vertices()

            for i, (m, n) in enumerate(self.graph.edges):
                max_vertices = max(num_vertices)
                # few number of vertices -> large capacity -> small cost -> cut here
                self.graph[m][n].update({'capacity': ((max_vertices - num_vertices[
                    i]) / max_vertices if normalise else max_vertices - num_vertices[i]) * factor})

        elif attribute == 'area_misalign':
            # area_misalign makes little sense as observed from the results
            logger.warning('attribute "area_misalign" is deprecated')

            # area of the mis-aligned regions from both cells
            for i, (m, n) in enumerate(self.graph.edges):
                # compute interface
                interface = cells[self._uid_to_index(m)].intersection(cells[self._uid_to_index(n)])
                if cache_interfaces:
                    self._cached_interfaces[m, n] = interface  # uid pair as key

                for facet_m in cells[self._uid_to_index(m)].facets():
                    for facet_n in cells[self._uid_to_index(n)].facets():
                        if facet_m.ambient_Hrepresentation()[0].A() == -facet_n.ambient_Hrepresentation()[0].A() and \
                                facet_m.ambient_Hrepresentation()[0].b() == -facet_n.ambient_Hrepresentation()[0].b():
                            # two facets coplanar
                            # area of the misalignment
                            if engine == 'Qhull':
                                area[i] = ConvexHull(
                                    facet_m.as_polyhedron().affine_hull_projection().vertices_list()).volume + ConvexHull(
                                    facet_n.as_polyhedron().affine_hull_projection().vertices_list()).volume - 2 * ConvexHull(
                                    interface.affine_hull_projection().vertices_list()).volume
                            else:
                                area[i] = RR(
                                    facet_m.as_polyhedron().affine_hull_projection().volume() +
                                    facet_n.as_polyhedron().affine_hull_projection().volume() -
                                    2 * interface.affine_hull_projection().volume())

            for i, (m, n) in enumerate(self.graph.edges):
                max_area = max(area)
                self.graph[m][n].update(
                    {'capacity': (area[i] / max_area if normalise else area[i]) * factor})

        elif attribute == 'volume_difference':
            # encourage partition between relatively a big cell and a small cell
            for i, (m, n) in enumerate(self.graph.edges):
                if engine == 'Qhull':
                    volume[i] = abs(ConvexHull(cells[self._uid_to_index(m)].vertices_list()).volume - ConvexHull(
                        cells[self._uid_to_index(n)].vertices_list()).volume) / max(
                        ConvexHull(cells[self._uid_to_index(m)].vertices_list()).volume,
                        ConvexHull(cells[self._uid_to_index(n)].vertices_list()).volume)
                else:
                    volume[i] = RR(
                        abs(cells[self._uid_to_index(m)].volume() - cells[self._uid_to_index(n)].volume()) / max(
                            cells[self._uid_to_index(m)].volume(), cells[self._uid_to_index(n)].volume()))

            for i, (m, n) in enumerate(self.graph.edges):
                max_volume = max(volume)
                # large difference -> large capacity -> small cost -> cut here
                self.graph[m][n].update(
                    {'capacity': (volume[i] / max_volume if normalise else volume[i]) * factor})

    def assign_weights_to_st_links(self, weights):
        """
        Assign weights to edges between each cell and the s-t nodes.

        Parameters
        ----------
        weights: dict
            Weights in respect to each node, can be the occupancy probability or the signed distance of the cells.
        """
        for i in self.uid:
            self.graph.add_edge(i, 's', capacity=weights[i])
            self.graph.add_edge(i, 't', capacity=1 - weights[i])  # make sure


    def extract_gt(self, cells, occ, filename):

        assert (len(cells) == occ.shape[0])

        tris = []
        points = []
        for e0, e1 in self.graph.edges:

            if e0 > e1:
                continue

            occ1 = occ[self._uid_to_index(e0)]
            occ2 = occ[self._uid_to_index(e1)]

            if occ1 != occ2:
                interface = cells[self._uid_to_index(e0)].intersection(cells[self._uid_to_index(e1)])

                verts = np.array(interface.vertices(),dtype=object)
                # verts = tuple(interface.vertices_list())
                correct_order = self._sorted_vertex_indices(interface.adjacency_matrix())
                # verts=self.orientFacet(verts[correct_order],outside)
                # tris.append(verts)
                verts = verts[correct_order]
                for i in range(verts.shape[0]):
                    points.append(tuple(verts[i,:]))
                tris.append(verts)



        pset = set(points)
        pset = np.array(list(pset),dtype=object)
        facets = []
        for tri in tris:
            face = []
            for p in tri:
                # face.append(np.argwhere((pset == p).all(-1))[0][0])
                face.append(np.argwhere((np.equal(pset,p,dtype=object)).all(-1))[0][0])
                # face.append(np.argwhere(np.isin(pset, p).all(-1))[0][0])
                # face.append(np.argwhere(np.isclose(pset, p,atol=tol*1.01).all(-1))[0][0])
            facets.append(face)

        self.pset = np.array(pset,dtype=float)
        self.facets = facets

        logger.debug('Save polygon mesh to {}'.format(filename))
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # self.toTrimesh(filename)
        # self.write_obj(filename)
        self.write_off(filename)
        # self.write_ply(filename)

        a = 4

    def extract_gt_cgal(self,cells,occ,filename):

        assert(len(cells)==occ.shape[0])

        points = []
        polygons_len = []
        polygons_index = []
        vcount = 0
        tris = []
        for e0,e1 in self.graph.edges:

            occ1 = occ[self._uid_to_index(e0)]
            occ2 = occ[self._uid_to_index(e1)]

            if occ1 != occ2:
                interface = cells[self._uid_to_index(e0)].intersection(cells[self._uid_to_index(e1)])

                poly = []
                for p in np.array(interface.vertices()):
                    points.append(p)
                    poly.append(vcount)
                    vcount += 1
                poly = np.array(poly)
                poly = poly[self._sorted_vertex_indices(interface.adjacency_matrix())]
                polygons_index.append(poly)
                polygons_len.append(len(interface.vertices()))

        points = np.array(points)
        polygons_len = np.array(polygons_len)
        polygons_index = np.concatenate(polygons_index, axis=0)
        logger.info('Save polygon mesh to {}'.format(filename))

        sm = s2m.Soup2Mesh()
        sm.loadSoup(points, polygons_len, polygons_index)
        triangulate = False
        sm.makeMesh(triangulate)
        sm.saveMesh(filename)


    def cut(self):
        """
        Perform cutting operation.

        Returns
        ----------
        cut_value: float
            Cost of the cut
        reachable: list of int
            Reachable nodes from the S node
        """
        tik = time.time()
        cut_value, partition = nx.algorithms.flow.minimum_cut(self.graph, 's', 't')
        reachable, non_reachable = partition
        reachable.remove('s')
        non_reachable.remove('t')
        self.reachable = reachable
        self.non_reachable = non_reachable

        logger.debug('cut performed: {:.2f} s'.format(time.time() - tik))
        logger.debug('cut_value: {:.2f}'.format(cut_value))
        logger.debug('number of extracted cells: {}'.format(len(reachable)))
        return cut_value, reachable

    @staticmethod
    def _sorted_vertex_indices(adjacency_matrix):
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



    def save_surface_obj(self, filepath, cells=None, engine='rendering'):
        """
        Save the outer surface to an OBJ file, from interfaces between cells being cut.

        Parameters
        ----------
        filepath: str or Path
            Filepath to save obj file
        cells: None or list of Polyhedra objects
            Polyhedra cells
        engine: str
            Engine to extract surface, can be 'rendering', 'sorting' or 'projection'
        """

        if not self.reachable:
            logger.error('no reachable cells. aborting')
            return
        elif not self.non_reachable:
            logger.error('no unreachable cells. aborting')
            return

        if not self._cached_interfaces and not cells:
            logger.error('neither cached interfaces nor cells are available. aborting')
            return

        if engine not in {'rendering', 'sorting', 'projection'}:
            logger.error('engine can be "rendering", "sorting" or "projection"')
            return

        surface = None
        surface_str = ''
        num_vertices = 0
        tik = time.time()

        for edge in self.graph.edges:
            # facet is where one cell being outside and the other one being inside
            if edge[0] in self.reachable and edge[1] in self.non_reachable:
                # retrieve interface and orient as on edge[0]
                if self._cached_interfaces:
                    interface = self._cached_interfaces[edge[0], edge[1]] if (edge[0],
                                                                              edge[1]) in self._cached_interfaces else \
                        self._cached_interfaces[edge[1], edge[0]]
                else:
                    interface = cells[self._uid_to_index(edge[0])].intersection(cells[self._uid_to_index(edge[1])])

            elif edge[1] in self.reachable and edge[0] in self.non_reachable:
                # retrieve interface and orient as on edge[1]
                if self._cached_interfaces:
                    interface = self._cached_interfaces[edge[1], edge[0]] if (edge[1],
                                                                              edge[0]) in self._cached_interfaces else \
                        self._cached_interfaces[edge[0], edge[1]]
                else:
                    interface = cells[self._uid_to_index(edge[1])].intersection(cells[self._uid_to_index(edge[0])])

            else:
                # where no cut is made
                continue

            if engine == 'rendering':
                surface += interface.render_solid()

            elif engine == 'sorting':
                for v in interface.vertices():
                    surface_str += 'v {} {} {}\n'.format(float(v[0]), float(v[1]), float(v[2]))
                vertex_indices = [i + num_vertices + 1 for i in
                                  self._sorted_vertex_indices(interface.adjacency_matrix())]
                surface_str += 'f ' + ' '.join([str(f) for f in vertex_indices]) + '\n'
                num_vertices += len(vertex_indices)

            elif engine == 'projection':
                projection = interface.projection()
                polygon = projection.polygons[0]
                for v in projection.coords:
                    surface_str += 'v {} {} {}\n'.format(float(v[0]), float(v[1]), float(v[2]))
                surface_str += 'f ' + ' '.join([str(f + num_vertices + 1) for f in polygon]) + '\n'
                num_vertices += len(polygon)

        if engine == 'rendering':
            surface_obj = surface.obj_repr(surface.default_render_params())

            for o in range(len(surface_obj)):
                surface_str += surface_obj[o][0] + '\n'
                surface_str += '\n'.join(surface_obj[o][2]) + '\n'
                surface_str += '\n'.join(surface_obj[o][3]) + '\n'  # contents[o][4] are the interior facets

        logger.info('surface extracted: {:.2f} s'.format(time.time() - tik))

        # create the dir if not exists
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            f.writelines(surface_str)


    def write_ply(self,filename):

        f = open(filename[:-3]+"ply", 'w')
        f.write("ply\nformat ascii 1.0\n")
        f.write("element vertex {}\n".format(self.pset.shape[0]))
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("element face {}\n".format(len(self.facets)))
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for p in self.pset:
            f.write("{} {} {}\n".format(p[0],p[1],p[2]))
        for face in self.facets:
            f.write("{}".format(len(face)))
            for v in face:
                f.write(" {}".format(v))
            f.write('\n')
        f.close()
        # f.write("primitives\n")

    def write_obj(self,filename):

        f = open(filename[:-3]+"obj",'w')
        # f.write("OFF\n")
        # f.write("{} {} 0\n".format(pset.shape[0],len(facets)))
        for p in self.pset:
            f.write("v {} {} {}\n".format(p[0],p[1],p[2]))
        for face in self.facets:
            f.write("f")
            for v in face:
                f.write(" {}".format(v+1))
            f.write('\n')
        f.close()

    def write_off(self,filename):

        f = open(filename[:-3]+"off",'w')
        f.write("OFF\n")
        f.write("{} {} 0\n".format(self.pset.shape[0],len(self.facets)))
        for p in self.pset:
            f.write("{} {} {}\n".format(p[0],p[1],p[2]))
        for face in self.facets:
            f.write("{}".format(len(face)))
            for v in face:
                f.write(" {}".format(v))
            f.write('\n')
        f.close()

    def toTrimesh(self, filename):

        # need first to PyMesh then to Trimesh
        # because trimesh can only do triangle facets
        mesh = pymesh.form_mesh(vertices, faces)
        mesh=trimesh.Trimesh(vertices=self.pset,faces=self.facets)
        mesh.repair.fix_normals
        mesh.export(filename)


    def orientFacet(self,verts,outside):

        a=verts[1]-verts[0]
        b=verts[2]-verts[0]
        c=outside-verts[0]
        if np.linalg.det(np.array([a,b,c])<0):
            return verts
        else:
            return np.flip(verts,axis=0)

    def unique_rows(self, A, atol=10e-5):
        """Get unique (within atol) rows of a 2D np.array A."""

        remove = np.zeros(A.shape[0], dtype=bool)  # Row indexes to be removed.
        for i in range(A.shape[0]):  # Not very optimized, but simple.
            equals = np.all(np.isclose(A[i, :], A[(i + 1):, :], atol=atol), axis=1)
            remove[(i + 1):] = np.logical_or(remove[(i + 1):], equals)
        return A[np.logical_not(remove)]

    def extract_surface_cgal(self, filename, cells=None):
        """
        Save the outer surface to an OBJ file, from interfaces between cells being cut.

        Parameters
        ----------
        filepath: str or Path
            Filepath to save obj file
        cells: None or list of Polyhedra objects
            Polyhedra cells
        engine: str
            Engine to extract surface, can be 'rendering', 'sorting' or 'projection'
        """

        if not self.reachable:
            logger.error('no reachable cells. aborting')
            return
        elif not self.non_reachable:
            logger.error('no unreachable cells. aborting')
            return

        points = []
        polygons_len = []
        polygons_index = []
        vcount = 0
        for edge in self.graph.edges:
            # facet is where one cell being outside and the other one being inside
            if edge[0] in self.reachable and edge[1] in self.non_reachable:
                # retrieve interface and orient as on edge[0]
                if self._cached_interfaces:
                    interface = self._cached_interfaces[edge[0], edge[1]] \
                        if (edge[0], edge[1]) in self._cached_interfaces \
                        else \
                        self._cached_interfaces[edge[1], edge[0]]
                else:
                    interface = cells[self._uid_to_index(edge[0])].intersection(cells[self._uid_to_index(edge[1])])

                outside = np.array(cells[self._uid_to_index(edge[0])].center())

            elif edge[1] in self.reachable and edge[0] in self.non_reachable:
                # retrieve interface and orient as on edge[1]
                if self._cached_interfaces:
                    interface = self._cached_interfaces[edge[1], edge[0]] \
                        if (edge[1], edge[0]) in self._cached_interfaces \
                        else \
                        self._cached_interfaces[edge[0], edge[1]]
                else:
                    interface = cells[self._uid_to_index(edge[1])].intersection(cells[self._uid_to_index(edge[0])])

                outside = np.array(cells[self._uid_to_index(edge[1])].center())

            else:
                # where no cut is made
                continue

            poly = []
            for p in np.array(interface.vertices()):
                points.append(p)
                poly.append(vcount)
                vcount+=1
            poly = np.array(poly)
            poly = poly[self._sorted_vertex_indices(interface.adjacency_matrix())]
            polygons_index.append(poly)
            polygons_len.append(len(interface.vertices()))

        points = np.array(points)
        polygons_len = np.array(polygons_len)
        polygons_index = np.concatenate(polygons_index,axis=0)
        logger.info('Save polygon mesh to {}'.format(filename))

        sm = s2m.Soup2Mesh()
        sm.loadSoup(points,polygons_len, polygons_index)
        triangulate=False
        sm.makeMesh(triangulate)
        sm.saveMesh(filename)



    def extract_surface(self, filename, cells=None):
        """
        Save the outer surface to an OBJ file, from interfaces between cells being cut.

        Parameters
        ----------
        filepath: str or Path
            Filepath to save obj file
        cells: None or list of Polyhedra objects
            Polyhedra cells
        engine: str
            Engine to extract surface, can be 'rendering', 'sorting' or 'projection'
        """

        if not self.reachable:
            logger.error('no reachable cells. aborting')
            return
        elif not self.non_reachable:
            logger.error('no unreachable cells. aborting')
            return

        interfaces=[]
        tris=[]
        for edge in self.graph.edges:
            # facet is where one cell being outside and the other one being inside
            if edge[0] in self.reachable and edge[1] in self.non_reachable:
                # retrieve interface and orient as on edge[0]
                if self._cached_interfaces:
                    interface = self._cached_interfaces[edge[0], edge[1]] \
                        if (edge[0],edge[1]) in self._cached_interfaces \
                        else \
                        self._cached_interfaces[edge[1], edge[0]]
                else:
                    interface = cells[self._uid_to_index(edge[0])].intersection(cells[self._uid_to_index(edge[1])])
                
                outside = np.array(cells[self._uid_to_index(edge[0])].center())

            elif edge[1] in self.reachable and edge[0] in self.non_reachable:
                # retrieve interface and orient as on edge[1]
                if self._cached_interfaces:
                    interface = self._cached_interfaces[edge[1], edge[0]] \
                        if (edge[1],edge[0]) in self._cached_interfaces \
                        else \
                        self._cached_interfaces[edge[0], edge[1]]
                else:
                    interface = cells[self._uid_to_index(edge[1])].intersection(cells[self._uid_to_index(edge[0])])
                
                outside = np.array(cells[self._uid_to_index(edge[1])].center())

            else:
                # where no cut is made
                continue

            interfaces.append(interface)
            verts=np.array(interface.vertices())
            correct_order=self._sorted_vertex_indices(interface.adjacency_matrix())
            # verts=self.orientFacet(verts[correct_order],outside)
            # tris.append(verts)
            tris.append(verts[correct_order])


        tol=0.05
        points = np.concatenate(tris, axis=0)
        pset = np.unique(points, axis=0)
        # pset = self.unique_rows(points, atol=tol)
        facets=[]
        for tri in tris:
            face = []
            for p in tri:
                face.append(np.argwhere(np.isin(pset, p).all(-1))[0][0])
                # face.append(np.argwhere(np.isclose(pset, p,atol=tol*1.01).all(-1))[0][0])
            facets.append(face)

        self.pset = pset
        self.facets = facets

        logger.debug('Save polygon mesh to {}'.format(filename))
        os.makedirs(os.path.dirname(filename),exist_ok=True)
        # self.toTrimesh(filename)
        # self.write_obj(filename)
        # self.write_off(filename)
        self.write_ply(filename)

        # TODO: color facets by primitive, by plane-equation (in fact Hrepresentation from sage) is already done, but not exactly what I want.
        # if facets have same plane equation and if they have common vertices, then they are from the same primitive





    def save_surface_obj_colored(self, filepath, cells=None, engine='rendering'):
        """
        Save the outer surface to an OBJ file, from interfaces between cells being cut.

        Parameters
        ----------
        filepath: str or Path
            Filepath to save obj file
        cells: None or list of Polyhedra objects
            Polyhedra cells
        engine: str
            Engine to extract surface, can be 'rendering', 'sorting' or 'projection'
        """

        if not self.reachable:
            logger.error('no reachable cells. aborting')
            return
        elif not self.non_reachable:
            logger.error('no unreachable cells. aborting')
            return

        if not self._cached_interfaces and not cells:
            logger.error('neither cached interfaces nor cells are available. aborting')
            return

        if engine not in {'rendering', 'sorting', 'projection'}:
            logger.error('engine can be "rendering", "sorting" or "projection"')
            return

        surface = None
        surface_str = ''
        num_vertices = 0
        tik = time.time()

        interfaces=[]
        interfaces_Hrep=[]
        for edge in self.graph.edges:
            # facet is where one cell being outside and the other one being inside
            if edge[0] in self.reachable and edge[1] in self.non_reachable:
                # retrieve interface and orient as on edge[0]
                if self._cached_interfaces:
                    interface = self._cached_interfaces[edge[0], edge[1]] if (edge[0],
                                                                              edge[1]) in self._cached_interfaces else \
                        self._cached_interfaces[edge[1], edge[0]]
                else:
                    interface = cells[self._uid_to_index(edge[0])].intersection(cells[self._uid_to_index(edge[1])])

            elif edge[1] in self.reachable and edge[0] in self.non_reachable:
                # retrieve interface and orient as on edge[1]
                if self._cached_interfaces:
                    interface = self._cached_interfaces[edge[1], edge[0]] if (edge[1],
                                                                              edge[0]) in self._cached_interfaces else \
                        self._cached_interfaces[edge[0], edge[1]]
                else:
                    interface = cells[self._uid_to_index(edge[1])].intersection(cells[self._uid_to_index(edge[0])])

            else:
                # where no cut is made
                continue


            interfaces_Hrep.append(np.array(interface.Hrepresentation()[0]))
            interfaces.append(interface)

        # if facets have same plane equation and if they have common vertices, then they are from the same primitive


        interfaces_Hrep = np.array(interfaces_Hrep,dtype=np.float64)
        interfaces_Hrep_set = np.unique(interfaces_Hrep,axis=0)
        colors = np.random.random(size=(interfaces_Hrep_set.shape[0],3))

        for i,interface in enumerate(interfaces):
            # get the color
            c=colors[np.argwhere(np.isin(interfaces_Hrep_set, interfaces_Hrep[i]))].flatten().flatten()
            if engine == 'rendering':
                surface += interface.render_solid()

            elif engine == 'sorting':
                for v in interface.vertices():
                    surface_str += 'v {} {} {} {} {} {}\n'.format(float(v[0]), float(v[1]), float(v[2]), float(c[0]), float(c[1]), float(c[2]))
                vertex_indices = [j + num_vertices + 1 for j in
                                  self._sorted_vertex_indices(interface.adjacency_matrix())]
                surface_str += 'f ' + ' '.join([str(f) for f in vertex_indices]) + '\n'
                num_vertices += len(vertex_indices)

            elif engine == 'projection':
                projection = interface.projection()
                polygon = projection.polygons[0]
                for v in projection.coords:
                    surface_str += 'v {} {} {}\n'.format(float(v[0]), float(v[1]), float(v[2]))
                surface_str += 'f ' + ' '.join([str(f + num_vertices + 1) for f in polygon]) + '\n'
                num_vertices += len(polygon)

        if engine == 'rendering':
            surface_obj = surface.obj_repr(surface.default_render_params())

            for o in range(len(surface_obj)):
                surface_str += surface_obj[o][0] + '\n'
                surface_str += '\n'.join(surface_obj[o][2]) + '\n'
                surface_str += '\n'.join(surface_obj[o][3]) + '\n'  # contents[o][4] are the interior facets

        logger.info('colored surface extracted: {:.2f} s'.format(time.time() - tik))

        # create the dir if not exists
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            f.writelines(surface_str)

    def draw(self):
        """
        Draw the graph with nodes represented by their UID.
        """
        import matplotlib.pyplot as plt
        plt.subplot(121)
        nx.draw(self.graph, with_labels=True, font_weight='bold')
        plt.show()

    def _uid_to_index(self, query):
        """
        Convert index in the node list to that in the cell list.

        The rationale behind is #nodes == #cells (when a primitive is settled down).

        Parameters
        ----------
        query: int
            Query uid in the node list

        Returns
        -------
        as_int: int
            Query uid in the cell list
        """
        return self.uid.index(query)

    def _index_to_uid(self, query):
        """
        Convert index to node UID.

        Parameters
        ----------
        query: int
            Query index in the node list

        Returns
        -------
        as_int: int
            Node UID
        """
        return self.uid[query]

    def _sort_uid(self):
        """
        Sort UIDs for graph structure loaded from an external file.

        Returns
        -------
        as_list: list of int
            Sorted UIDs
        """
        return sorted([int(i) for i in self.graph.nodes])

    def to_indices(self, uids):
        """
        Convert UIDs to indices.

        Parameters
        ----------
        uids: list of int
            UIDs of nodes

        Returns
        -------
        as_list: list of int
            Indices of nodes
        """
        return [self._uid_to_index(i) for i in uids]

    def to_dict(self, weights_list):
        """
        Convert a weight list to weight dict keyed by self.uid.

        Parameters
        ----------
        weights_list: list of
            Weight list

        Returns
        -------
        as_dict: dict
            Weight dict
        """
        return {self.uid[i]: weight for i, weight in enumerate(weights_list)}
