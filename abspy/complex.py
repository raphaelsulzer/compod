"""
complex.py
----------

Cell complex from planar primitive arrangement.

A linear cell complex is constructed from planar primitives
with adaptive binary space partitioning: upon insertion of a primitive
only the local cells that are intersecting it will be updated,
so will be the corresponding adjacency graph of the complex.
"""

import os
import string
from pathlib import Path
import itertools
import heapq
from copy import copy
from random import random, choices, uniform
import pickle
import time
import multiprocessing
from fractions import Fraction
from copy import deepcopy
import numpy as np
from tqdm import trange
import networkx as nx
import trimesh
from sage.all import polytopes, QQ, RR, Polyhedron
from .logger import attach_to_log
import matplotlib.pyplot as plt

from copy import  deepcopy


import sys
sys.path.append("/home/rsulzer/cpp/compact_mesh_reconstruction/build/release/Benchmark/PyLabeler")
import libPyLabeler as PL

logger = attach_to_log()

from treelib import Tree

class CellComplex:
    """
    Class of cell complex from planar primitive arrangement.
    """
    def __init__(self, planes, halfspaces, bounds, points=None, initial_bound=None, initial_padding=0.1, additional_planes=None,
                 build_graph=False, quiet=False,exporter=None):
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

        self.exporter = exporter


        self.quiet = quiet
        if self.quiet:
            logger.disabled = True

        logger.debug('Init cell complex with padding {}'.format(initial_padding))

        self.bounds = bounds  # numpy.array over RDF
        self.planes = planes  # numpy.array over RDF
        self.halfspaces = halfspaces
        self.points = points

        # missing planes due to occlusion or incapacity of RANSAC
        self.additional_planes = additional_planes



        # self.initial_bound = initial_bound if initial_bound else self._pad_bound(
        #     [np.amin(bounds[:, 0, :], axis=0), np.amax(bounds[:, 1, :], axis=0)],
        #     padding=initial_padding)

        # self.initial_bound = initial_bound if initial_bound else self._my_pad_bound(
        #     bounds,
        #     padding=initial_padding)

        # self.cells = [self._construct_initial_cell()]  # list of QQ
        # self.cells_bounds = [self.cells[0].bounding_box()]  # list of QQ

        if build_graph:
            self.graph = nx.Graph()
            self.graph.add_node(0)  # the initial cell
            self.index_node = 0  # unique for every cell ever generated
        else:
            self.graph = None

        self.constructed = False

    def _construct_initial_cell(self):
        """
        Construct initial bounding cell.

        Return
        ----------
        as_object: Polyhedron object
            Polyhedron object of the initial cell,
            a cuboid with 12 triangular facets.
        """
        return polytopes.cube(
            intervals=[[QQ(self.initial_bound[0][i]), QQ(self.initial_bound[1][i])] for i in range(3)])


    def get_bounding_box(self,m,scale=1.2):

        self.bounding_verts = []
        # points = np.load(m["pointcloud"])["points"]
        points = np.load(m["occ"])["points"]

        ppmin = points.min(axis=0)
        ppmax = points.max(axis=0)

        # ppmin = [-40,-40,-40]
        # ppmax = [40,40,40]

        pmin=[]
        for p in ppmin:
            pmin.append(Fraction(str(p)))
        pmax=[]
        for p in ppmax:
            pmax.append(Fraction(str(p)))

        self.bounding_verts.append(pmin)
        self.bounding_verts.append([pmin[0],pmax[1],pmin[2]])
        self.bounding_verts.append([pmin[0],pmin[1],pmax[2]])
        self.bounding_verts.append([pmin[0],pmax[1],pmax[2]])
        self.bounding_verts.append(pmax)
        self.bounding_verts.append([pmax[0],pmin[1],pmax[2]])
        self.bounding_verts.append([pmax[0],pmax[1],pmin[2]])
        self.bounding_verts.append([pmax[0],pmin[1],pmin[2]])

        self.bounding_poly = Polyhedron(vertices=self.bounding_verts)

    def write_graph(self, m, graph, subfolder="", color = None):

        c = color if color is not None else np.random.random(size=3)
        c = (c*255).astype(int)

        path = os.path.join(os.path.dirname(m['planes']),subfolder)
        os.makedirs(path,exist_ok=True)
        filename = os.path.join(path,'graph.obj')

        edge_strings = []
        f = open(filename,'w')
        all_nodes = np.array(graph.nodes())
        for i,node in enumerate(graph.nodes(data=True)):
            centroid = np.array(node[1]["convex"].centroid())
            f.write("v {} {} {} {} {} {}\n".format(centroid[0],centroid[1],centroid[2],c[0],c[1],c[2]))
            edges = list(graph.edges(node[0]))
            for c1,c2 in edges:
                nc1 = np.where(all_nodes==c1)[0][0]
                nc2 = np.where(all_nodes==c2)[0][0]
                edge_strings.append("l {} {}\n".format(nc1+1,nc2+1))


        for edge in edge_strings:
            f.write(edge)

        f.close()

        a=4





    def write_cells(self, m, polyhedron, points=None, subfolder="partitions",count=0, color=None):

        c = color if color is not None else np.random.random(size=3)
        c = (c*255).astype(int)

        path = os.path.join(os.path.dirname(m['planes']),subfolder)
        os.makedirs(path,exist_ok=True)
        filename = os.path.join(path,str(count)+'.obj')

        ss = polyhedron.render_solid().obj_repr(polyhedron.render_solid().default_render_params())

        f = open(filename,'w')

        verts = ss[2]
        for v in verts:
            f.write(v + " {} {} {}\n".format(c[0],c[1],c[2]))
        faces = ss[3]
        for fa in faces:
            f.write(fa+"\n")

        # for out in ss[2:4]:
        #     for line in out:
        #         f.write(line+"\n")

        if points is not None:
            for p in points:
                f.write("v {} {} {} {} {} {}\n".format(p[0],p[1],p[2],c[0],c[1],c[2]))

        f.close()

    def write_faces(self,m,facet,subfolder="facets",count=0, color=None):

        c = color if color is not None else np.random.random(size=3)
        c = (c*255).astype(int)

        path = os.path.join(os.path.dirname(m['planes']),subfolder)
        os.makedirs(path,exist_ok=True)
        filename = os.path.join(path,str(count)+'.obj')

        ss = facet.render_solid().obj_repr(facet.render_solid().default_render_params())

        f = open(filename,'w')
        verts = ss[2]
        for v in verts:
            f.write(v + " {} {} {}\n".format(c[0],c[1],c[2]))
        faces = ss[3]
        for fa in faces:
            f.write(fa+"\n")

        f.close()

    def write_points(self,m,points,filename="points",count=0, color=None):

        path = os.path.join(os.path.dirname(m['planes']))
        filename = os.path.join(path,'{}.off'.format(filename))

        f = open(filename, 'w')
        f.write("OFF\n")
        f.write("{} 0 0\n".format(points.shape[0]))
        for p in points:
            f.write("{} {} {}\n".format(p[0],p[1],p[2]))
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


    def _contains(self,polyhedron,points):
        """check if any of the points are contained in the polyhedron"""

        ineqs = np.array(polyhedron.inequalities())
        # careful here, the ineqs from SageMath have a strange order
        inside = points[:, 0] * ineqs[:, 1, np.newaxis] + points[:, 1] * ineqs[:, 2, np.newaxis] + \
                  points[:, 2] * ineqs[:, 3, np.newaxis] + ineqs[:, 0, np.newaxis]

        inside = (np.sign(inside)+1).astype(bool)

        points_inside_all_planes = inside.all(axis=0)
        at_least_one_point_inside_all_planes = points_inside_all_planes.any()

        return at_least_one_point_inside_all_planes





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

    def extract_gt(self, filename):

        tris = []
        points = []
        for e0, e1 in self.graph.edges:

            if e0 > e1:
                continue

            c0 = self.graph.nodes[e0]
            c1 = self.graph.nodes[e1]

            if c0["occupancy"] != c1["occupancy"]:

                interface = self.graph[e0][e1]["intersection"]
                interface = interface if interface is not None else c0["convex"].intersection(c1["convex"])

                verts = np.array(interface.vertices(),dtype=object)
                correct_order = self._sorted_vertex_indices(interface.adjacency_matrix())
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

    def label_graph_nodes(self, m, n_test_points=50,export=False):


        pl=PL.PyLabeler(n_test_points)
        pl.loadMesh(m["mesh"])
        points = []
        points_len = []
        for i,node in enumerate(self.graph.nodes(data=True)):
            cell = node[1]['convex']
            if export:
                self.write_cells(m,cell,count=i,subfolder="final_cells")
            pts = np.array(cell.vertices())
            points.append(pts)
            # print(pts)
            points_len.append(pts.shape[0])


        # assert(isinstance(points[0].dtype,np.float32))
        occs = pl.labelCells(np.array(points_len),np.concatenate(points,axis=0))
        del pl

        for i,node in enumerate(self.graph.nodes(data=True)):
            node[1]["occupancy"] = np.rint(occs[i]).astype(int)
            if export:
                col = [1,0,0] if node[1]["occupancy"] == 1 else [0,0,1]
                self.write_cells(m,node[1]['convex'],count=i,subfolder="in_out_cells",color=np.array(col))

        if export:
            self.write_graph(m,self.graph)



    def _get_best_plane(self,current_ids,planes,point_groups,export=False):

        ### the whole thing vectorized. doesn't really work for some reason
        # UPDATE: should work, first tries where with wrong condition
        # planes = np.repeat(vertex_group.planes[np.newaxis,current_ids,:],current_ids.shape[0],axis=0)
        # pgs = np.repeat(point_groups[np.newaxis,current_ids,:,:],current_ids.shape[0],axis=0)
        #
        # which_side = planes[:,:,0,np.newaxis] * pgs[:,:,:,0] + planes[:,:,1,np.newaxis] * pgs[:,:,:,1] + planes[:,:,2,np.newaxis] * pgs[:,:,:,2] + planes[:,:,3,np.newaxis]

        ### find the plane which seperates all other planes without splitting them
        left_right = []
        for i,id in enumerate(current_ids):
            left = 0; right = 0
            for id2 in current_ids:
                if id == id2: continue
                which_side = planes[id, 0] * point_groups[id2][:, 0] + planes[id, 1] * point_groups[id2][:,1] + planes[id, 2] * point_groups[id2][:, 2] + planes[id, 3]
                left+= (which_side < 0).all(axis=-1)  ### check for how many planes all points of these planes fall on the left of the current plane
                right+= (which_side > 0).all(axis=-1)  ### check for how many planes all points of these planes fall on the right of the current plane
            if left == current_ids.shape[0]-1 or right == current_ids.shape[0]-1:
                return i

            left_right.append([left, right])

        assert(len(left_right)==len(current_ids))

        left_right = np.array(left_right)
        left_right = left_right.sum(axis=1)
        best_plane_id = np.argmax(left_right)


        return best_plane_id

    def _split_planes(self,best_plane_id,current_ids,planes,halfspaces,point_groups, th=1):

        '''
        :param best_plane_id:
        :param current_ids:
        :param planes:
        :param point_groups: padded 2d array of point groups with NaNs
        :param n_points_per_plane: real number of points per group (ie plane)
        :return: left and right planes
        '''

        best_plane = planes[current_ids[best_plane_id]]

        ### now put the planes into the left and right subspace of the best_plane split
        ### planes that lie in both subspaces are split (ie their point_groups are split) and appended as new planes to the planes array, and added to both subspaces
        left_planes = []
        right_planes = []
        for id in current_ids:

            if id == current_ids[best_plane_id]:
                continue

            which_side = best_plane[0] * point_groups[id][:, 0] + best_plane[1] * point_groups[id][:, 1] + best_plane[2] * point_groups[id][:, 2] + best_plane[3]

            left_points = point_groups[id][which_side < 0, :]
            right_points = point_groups[id][which_side > 0, :]

            assert (point_groups[id].shape[0] > th)  # threshold cannot be bigger than the detection threshold

            if (point_groups[id].shape[0] - left_points.shape[0]) < th:
                left_planes.append(id)
                point_groups[id] = left_points  # update the point group, in case some points got dropped according to threshold
            elif(point_groups[id].shape[0] - right_points.shape[0]) < th:
                right_planes.append(id)
                point_groups[id] = right_points # update the point group, in case some points got dropped according to threshold
            else:
                # print("id:{}: total-left/right: {}-{}/{}".format(current_ids[best_plane_id],n_points_per_plane[id],left_points.shape[0],right_points.shape[0]))
                if (left_points.shape[0] > th):
                    left_planes.append(planes.shape[0])
                    point_groups.append(left_points)
                    planes = np.vstack((planes, planes[id]))
                    halfspaces = np.vstack((halfspaces,halfspaces[id]))
                if (right_points.shape[0] > th):
                    right_planes.append(planes.shape[0])
                    point_groups.append(right_points)
                    planes = np.vstack((planes, planes[id]))
                    halfspaces = np.vstack((halfspaces,halfspaces[id]))

                self.split_count+=1

                planes[id, :] = np.nan
                point_groups[id][:, :] = np.nan

        return left_planes,right_planes, planes, halfspaces, point_groups




    def my_construct(self, m, mode=Tree.DEPTH, th=1, ordering="optimal", export=False):

        self.split_count = 0

        ## Tree.DEPTH seems slightly faster then Tree.WIDTH

        # TODO: i need a secomd ordering for when two planes have the same surface split score, take the one with the bigger area.
        # because randommly shuffling the planes before this function has a big influence on the result

        self.get_bounding_box(m)

        ### pad the point groups with NaNs to make a numpy array from the variable lenght list
        ### could maybe better be done with scipy sparse, but would require to rewrite the _get and _split functions used below

        ### make a new planes array, to which planes that are split can be appanded
        planes = deepcopy(self.planes)
        halfspaces = deepcopy(self.halfspaces)
        point_groups = list(self.points)

        cell_count = 0

        dim0points = []
        dim1points = []

        ## init the graph
        graph = nx.Graph()
        graph.add_node(cell_count, convex=self.bounding_poly)

        ## expand the tree as long as there is at least one plane in any of the subspaces
        tree = Tree()
        dd = {"convex": self.bounding_poly, "plane_ids": np.arange(self.planes.shape[0])}
        tree.create_node(tag=cell_count, identifier=cell_count, data=dd)  # root node
        children = tree.expand_tree(0, filter=lambda x: x.data["plane_ids"].shape[0], mode=mode)
        plane_count = 0
        for child in children:

            current_ids = tree[child].data["plane_ids"]

            ### get the best plane
            if ordering == "optimal":
                best_plane_id = self._get_best_plane(current_ids,planes,point_groups)
            else:
                best_plane_id = 0
            best_plane = planes[current_ids[best_plane_id]]
            plane_count+=1

            if current_ids[best_plane_id] >= len(self.planes):
                a=5

            ### export best plane
            if export:
                color = [1, 0, 0] if current_ids[best_plane_id] >= len(self.planes) else [0, 1, 0]  # split planes are red, unsplit planes are green
                epoints = point_groups[current_ids[best_plane_id]]
                epoints = epoints[~np.isnan(epoints).all(axis=-1)]
                if epoints.shape[0]>3:
                    self.exporter.export_plane(os.path.dirname(m["planes"]), best_plane, epoints,count=str(plane_count),color=color)

            ### split the planes
            left_planes, right_planes, planes, halfspaces, point_groups, = self._split_planes(best_plane_id,current_ids,planes,halfspaces,point_groups, th)

            ## create the new convexes
            current_cell = tree[child].data["convex"]
            # hspace_positive, hspace_negative = [Polyhedron(ieqs=[inequality]) for inequality in
            #                                     self._inequalities(best_plane)]
            hspace_positive, hspace_negative = halfspaces[current_ids[best_plane_id],0], halfspaces[current_ids[best_plane_id],1]

            cell_negative = current_cell.intersection(hspace_negative)
            cell_positive = current_cell.intersection(hspace_positive)


            ## update tree by creating the new nodes with the planes that fall into it
            ## and update graph with new nodes
            if(cell_negative.dim() > 0):
                if export:
                    self.write_cells(m,cell_negative,count=str(cell_count+1)+"n")
                dd = {"convex": cell_negative,"plane_ids": np.array(left_planes)}
                cell_count = cell_count+1
                neg_cell_count = cell_count
                tree.create_node(tag=cell_count, identifier=cell_count, data=dd, parent=tree[child].identifier)
                graph.add_node(neg_cell_count,convex=cell_negative)

            if(cell_positive.dim() > 0):
                if export:
                    self.write_cells(m,cell_positive,count=str(cell_count+1)+"p")
                dd = {"convex": cell_positive,"plane_ids": np.array(right_planes)}
                cell_count = cell_count+1
                pos_cell_count = cell_count
                tree.create_node(tag=cell_count, identifier=cell_count, data=dd, parent=tree[child].identifier)
                graph.add_node(pos_cell_count,convex=cell_positive)

            if(cell_positive.dim() > 0 and cell_negative.dim() > 0):
                graph.add_edge(cell_count-1, cell_count, intersection=None)
                if export:
                    intersection = cell_negative.intersection(cell_positive)
                    self.write_faces(m,intersection,count=plane_count)

            ## add edges to other cells, these must be neigbors of the parent (her named child) of the new subspaces
            neighbors = list(graph[child])
            for n in neighbors:
                nconvex = graph.nodes[n]["convex"]

                negative_intersection = nconvex.intersection(cell_negative)
                if negative_intersection.dim() == 2:
                    graph.add_edge(n,neg_cell_count,intersection=negative_intersection)
                # elif negative_intersection.dim() == 1:
                #     dim1points.append(np.array(negative_intersection.vertices()))
                # elif negative_intersection.dim() == 0:
                #     dim0points.append(np.array(negative_intersection.vertices()))

                positive_intersection = nconvex.intersection(cell_positive)
                if positive_intersection.dim() == 2:
                    graph.add_edge(n, pos_cell_count, intersection=positive_intersection)
                # elif positive_intersection.dim() == 1:
                #     dim1points.append(np.array(positive_intersection.vertices()))
                # elif positive_intersection.dim() == 0:
                #     dim0points.append(np.array(positive_intersection.vertices()))



            ## remove the parent node
            graph.remove_node(child)

            # nx.draw(graph,with_labels=True)  # networkx draw()
            # plt.draw()
            # plt.show()

            # self.write_graph(m,graph)
            # self.cells = list(nx.get_node_attributes(graph, "convex").values())
            # self.save_obj(os.path.join(m["abspy"]["partition"]))



        if len(dim0points) > 0:
            self.write_points(m,np.concatenate(dim0points),"dim0points")
        if len(dim1points) > 0:
            self.write_points(m,np.concatenate(dim1points),"dim1points")

        self.graph = graph
        self.cells = list(nx.get_node_attributes(graph, "convex").values())
        self.constructed = True

        # ### reorder the planes and recalculate the bounds from the new point groups (ie the planes that were split)
        # self.planes = planes[plane_order]
        # self.points = []
        # self.bounds = []
        # point_groups = point_groups[plane_order]
        # for i,group in enumerate(point_groups):
        #     group = group[~np.isnan(group).all(axis=-1)]
        #     self.points.append(group)
        #     self.bounds.append(np.array([np.amin(group, axis=0), np.amax(group, axis=0)]))
        # self.points = np.array(self.points, dtype=object)
        # self.bounds = np.array(self.bounds)


        logger.info("Out of {} planes {} were split, making a total of {} planes now".format(len(self.planes),self.split_count,len(self.planes)+self.split_count))

        return 0



    def prioritise_planes(self, mode = ["vertical", "random"]):
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
        logger.info('prioritising planar primitives')

        # compute the priority
        indices_sorted_planes = self._sort_planes()


        if mode == "random":
            np.random.shuffle(indices_sorted_planes)
            indices_priority = indices_sorted_planes



        if mode == "vertical":
            indices_vertical_planes = self._vertical_planes(slope_threshold=0.9)
            bool_vertical_planes = np.in1d(indices_sorted_planes, indices_vertical_planes)
            indices_priority = np.append(indices_sorted_planes[bool_vertical_planes],
                                         indices_sorted_planes[np.invert(bool_vertical_planes)])

        # reorder both the planes and their bounds
        self.planes = self.planes[indices_priority]
        self.bounds = self.bounds[indices_priority]
        self.points = self.points[indices_priority]

        # append additional planes with highest priority
        if self.additional_planes is not None:
            self.planes = np.concatenate([self.additional_planes, self.planes], axis=0)
            additional_bounds = [[[-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf]]] * len(self.additional_planes)
            self.bounds = np.concatenate([additional_bounds, self.bounds], axis=0)  # never miss an intersection

        logger.debug('ordered planes: {}'.format(self.planes))
        logger.debug('ordered bounds: {}'.format(self.bounds))

    def _vertical_planes(self, slope_threshold=0.9, epsilon=10e-5):
        """
        Return vertical planes.

        Parameters
        ----------
        slope_threshold: float
            Slope threshold, above which the planes are considered vertical
        epsilon: float
            Trivial term to avoid NaN

        Returns
        -------
        as_int: (n,) int
            Indices of the vertical planar primitives
        """
        slope_squared = (self.planes[:, 0] ** 2 + self.planes[:, 1] ** 2) / (self.planes[:, 2] ** 2 + epsilon)
        return np.where(slope_squared > slope_threshold ** 2)[0]

    def _sort_planes(self, mode='norm'):
        """
        Sort planes.

        Parameters
        ----------
        mode: str
            Mode for sorting, can be 'volume' or 'norm'

        Returns
        -------
        as_int: (n,) int
            Indices by which the planar primitives are sorted based on their bounding box volume
        """
        if mode == 'volume':
            volume = np.prod(self.bounds[:, 1, :] - self.bounds[:, 0, :], axis=1)
            return np.argsort(volume)[::-1]
        elif mode == 'norm':
            sizes = np.linalg.norm(self.bounds[:, 1, :] - self.bounds[:, 0, :], ord=2, axis=1)
            return np.argsort(sizes)[::-1]
        elif mode == 'area':
            # project the points supporting each plane onto the plane
            # https://stackoverflow.com/questions/9605556/how-to-project-a-point-onto-a-plane-in-3d
            raise NotImplementedError
        else:
            raise ValueError('mode has to be "volume" or "norm"')

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
            self.graph.add_edge(self.index_node + 1, n)
            interface_negative = cell_negative.intersection(cell_neighbour)
            if interface_negative.dim() == 2:
                self.graph.add_edge(self.index_node + 2, n)
        else:
            # this neighbour must otherwise connect with the other child
            self.graph.add_edge(self.index_node + 2, n)

    def construct(self, exhaustive=False, num_workers=0):
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
        if exhaustive:
            logger.info('construct exhaustive cell complex'.format())
        else:
            logger.info('construct cell complex'.format())

        tik = time.time()

        pool = None
        if num_workers > 0:
            pool = multiprocessing.Pool(processes=num_workers)

        pbar = range(len(self.bounds)) if self.quiet else trange(len(self.bounds))
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

                    # append the edge in between
                    self.graph.add_edge(self.index_node + 1, self.index_node + 2)

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
                            kwargs.append({'n': n, 'positive': cell_positive, 'negative': cell_negative, 'neighbour': cell})

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
                    self.graph.remove_node(list(self.graph.nodes)[index_parent])

        self.constructed = True
        logger.debug('cell complex constructed: {:.2f} s'.format(time.time() - tik))


    @property
    def num_cells(self):
        """
        Number of cells in the complex.
        """
        return len(self.cells)

    @property
    def num_planes(self):
        """
        Number of planes in the complex, excluding the initial bounding box.
        """
        return len(self.planes)

    def volumes(self, multiplier=1.0, engine='Qhull'):
        """
        list of cell volumes.

        Parameters
        ----------
        multiplier: float
            Multiplier to the volume
        engine: str
            Engine to compute volumes, can be 'Qhull' or 'Sage' with native SageMath

        Returns
        -------
        as_float: list of float
            Volumes of cells
        """
        if engine == 'Qhull':
            from scipy.spatial import ConvexHull
            volumes = [None for _ in range(len(self.cells))]
            for i, cell in enumerate(self.cells):
                try:
                    volumes[i] = ConvexHull(cell.vertices_list()).volume * multiplier
                except:
                    # degenerate floating-point
                    volumes[i] = RR(cell.volume()) * multiplier
            return volumes

        elif engine == 'Sage':
            return [RR(cell.volume()) * multiplier for cell in self.cells]

        else:
            raise ValueError('engine must be either "Qhull" or "Sage"')

    def cell_representatives(self, location='center', num=1):
        """
        Return representatives of cells in the complex.

        Parameters
        ----------
        location: str
            'center' represents the average of the vertices of the polyhedron,
            'centroid' represents the center of mass/volume,
            'random' represents random point(s),
            'star' represents star-like point(s)
        num: int
            number of samples per cell, only applies to 'random' and 'star'

        Returns
        -------
        as_float: (n, 3) float for 'center' and 'centroid', or (m, n, 3) for 'random' and 'star'
            Representatives of cells in the complex.
        """
        if location == 'center':
            return [cell.center() for cell in self.cells]
        elif location == 'centroid':
            return [cell.centroid() for cell in self.cells]
        elif location == 'random':
            points = []
            for cell in self.cells:
                bbox = cell.bounding_box()
                points_cell = []
                while len(points_cell) < num:
                    sample = (uniform(bbox[0][0], bbox[1][0]), uniform(bbox[0][1], bbox[1][1]),
                              uniform(bbox[0][2], bbox[1][2]))
                    if cell.contains(sample):
                        points_cell.append(sample)
                points.append(points_cell)
            return points

        elif location == 'star':
            points = []
            for cell in self.cells:
                vertices = cell.vertices_list()
                if num <= len(vertices):
                    # vertices given high priority
                    points.append(choices(vertices, k=num))
                else:
                    num_per_vertex = num // len(vertices)
                    num_remainder = num % len(vertices)
                    centroid = cell.centroid()
                    points_cell = []
                    for vertex in vertices[:-1]:
                        points_cell.extend([vertex + (centroid - np.array(vertex)) / num_per_vertex * i
                                           for i in range(num_per_vertex)])
                    # last vertex consumes remainder points
                    points_cell.extend([vertices[-1] + (centroid - np.array(vertices[-1])) / (num_remainder + num_per_vertex)
                                       * i for i in range(num_remainder + num_per_vertex)])
                    points.append(points_cell)
            return points

        else:
            raise ValueError("expected 'center', 'centroid', 'random' or 'star' as mode, got {}".format(location))

    def cells_in_mesh(self, filepath_mesh, engine='ray'):
        """
        Return indices of cells that are inside a reference mesh.

        Parameters
        ----------
        filepath_mesh: str or Path
            Filepath to reference mesh
        engine: str
            Engine to compute predicate, can be 'ray' for ray intersection, or 'distance' for signed distance

        Returns
        -------
        as_int: (n, ) int
            Indices of cells being inside the reference mesh
        """
        mesh = trimesh.load_mesh(filepath_mesh)
        centers = self.cell_representatives(location='center')

        if engine == 'ray':
            # raytracing not stable for non-watertight mesh (incl. with inner structure)
            try:
                # https://trimsh.org/trimesh.ray.ray_pyembree.html
                contains = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh).contains_points(centers)
            except ModuleNotFoundError:
                # https://trimsh.org/trimesh.ray.ray_triangle.html
                logger.warning('pyembree installation not found; fall back to ray_triangle')
                contains = mesh.contains(centers)
            return contains.nonzero()[0]

        elif engine == 'distance':
            # https://trimsh.org/trimesh.proximity.html
            distances = trimesh.proximity.signed_distance(mesh, centers)
            return (distances >= 0).nonzero()[0]
        else:
            raise ValueError("expected 'ray' or 'distance' as engine, got {}".format(engine))

    def print_info(self):
        """
        Print info to console.
        """
        logger.info('number of planes: {}'.format(self.num_planes))
        logger.info('number of cells: {}'.format(self.num_cells))

    def save(self, filepath):
        """
        Save the cell complex to a CC file.

        Parameters
        ----------
        filepath: str or Path
            Filepath to save CC file, '.cc' suffix recommended
        """
        if self.constructed:
            # create the dir if not exists
            with open(filepath, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        else:
            raise RuntimeError('cell complex has not been constructed')

    def save_npy(self, filepath):
        """
        Save the cells to an npy file (deprecated).

        Parameters
        ----------
        filepath: str or Path
            Filepath to save npy file
        """
        if self.constructed:
            # create the dir if not exists
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            np.save(filepath, self.cells, allow_pickle=True)
        else:
            raise RuntimeError('cell complex has not been constructed')

    @staticmethod
    def _obj_str(cells, use_mtl=False, filename_mtl='colours.mtl'):
        """
        Convert a list of cells into a string of obj format.

        Parameters
        ----------
        cells: list of Polyhedra objects
            Polyhedra cells
        use_mtl: bool
            Use mtl attribute in obj if set True
        filename_mtl: None or str
            Material filename

        Returns
        -------
        scene_str: str
            String representation of the object
        material_str: str
            String representation of the material
        """
        scene = None
        for cell in cells:
            scene += cell.render_solid()

        # directly save the obj string from scene.obj() will bring the inverted facets
        scene_obj = scene.obj_repr(scene.default_render_params())
        if len(cells) == 1:
            scene_obj = [scene_obj]
        scene_str = ''
        material_str = ''

        if use_mtl:
            scene_str += f'mtllib {filename_mtl}\n'

        for o in range(len(cells)):
            scene_str += scene_obj[o][0] + '\n'

            if use_mtl:
                scene_str += scene_obj[o][1] + '\n'
                material_str += 'newmtl ' + scene_obj[o][1].split()[1] + '\n'
                material_str += 'Kd {:.3f} {:.3f} {:.3f}\n'.format(random(), random(), random())  # diffuse colour

            scene_str += '\n'.join(scene_obj[o][2]) + '\n'
            scene_str += '\n'.join(scene_obj[o][3]) + '\n'  # contents[o][4] are the interior facets
        return scene_str, material_str

    def save_obj(self, filepath, indices_cells=None, use_mtl=False):
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
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        cells = [self.cells[i] for i in indices_cells] if indices_cells is not None else self.cells
        scene_str, material_str = self._obj_str(cells, use_mtl=use_mtl, filename_mtl=f'{filepath.stem}.mtl')

        with open(filepath, 'w') as f:
            f.writelines("# cells: {}\n".format(len(self.cells)))
            f.writelines(scene_str)
        if use_mtl:
            with open(filepath.with_name(f'{filepath.stem}.mtl'), 'w') as f:
                f.writelines(material_str)

