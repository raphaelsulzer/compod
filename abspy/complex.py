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
from export import Exporter
from .logger import attach_to_log
from .primitive import VertexGroup
import csv

from skspatial.objects import Plane



logger = attach_to_log()

from treelib import Node, Tree

class CellComplex:
    """
    Class of cell complex from planar primitive arrangement.
    """
    def __init__(self, planes, bounds, points=None, initial_bound=None, initial_padding=0.1, additional_planes=None,
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
        self.points = points

        # missing planes due to occlusion or incapacity of RANSAC
        self.additional_planes = additional_planes



        self.initial_bound = initial_bound if initial_bound else self._pad_bound(
            [np.amin(bounds[:, 0, :], axis=0), np.amax(bounds[:, 1, :], axis=0)],
            padding=initial_padding)

        # self.initial_bound = initial_bound if initial_bound else self._my_pad_bound(
        #     bounds,
        #     padding=initial_padding)

        self.cells = [self._construct_initial_cell()]  # list of QQ
        self.cells_bounds = [self.cells[0].bounding_box()]  # list of QQ

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


    def get_bounding_box(self,m):

        self.bounding_verts = []
        # points = np.load(m["pointcloud"])["points"]
        points = np.load(m["occ"])["points"]

        ppmin = points.min(axis=0)
        ppmax = points.max(axis=0)

        ppmin = [-40,-40,-40]
        ppmax = [40,40,40]

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


    def write_cells(self, m, polyhedron, points=None):

        c = np.random.random(size=3)

        path = os.path.join(os.path.dirname(m['abspy']['partition']),"partitions")
        os.makedirs(path,exist_ok=True)
        filename = os.path.join(path,str(self.split_count)+'.obj')

        ss = polyhedron.render_solid().obj_repr(polyhedron.render_solid().default_render_params())

        f = open(filename,'w')
        for out in ss[2:4]:
            for line in out:
                f.write(line+"\n")

        if points is not None:
            for p in points:
                f.write("v {} {} {} {} {} {}\n".format(p[0],p[1],p[2],c[0],c[1],c[2]))

        f.close()

    def write_faces(self,m,facet):

        c = np.random.random(size=3)

        path = os.path.join(os.path.dirname(m['abspy']['partition']),"facets")
        os.makedirs(path,exist_ok=True)
        filename = os.path.join(path,str(self.face_count)+'.obj')

        ss = facet.render_solid().obj_repr(facet.render_solid().default_render_params())

        f = open(filename,'w')
        for out in ss[2:4]:
            for line in out:
                f.write(line+"\n")

        f.close()


    def contains(self,polyhedron,points):
        """check if any of the points are contained in the polyhedron"""

        ineqs = np.array(polyhedron.inequalities())
        # careful here, the ineqs from SageMath have a strange order
        inside = points[:, 0] * ineqs[:, 1, np.newaxis] + points[:, 1] * ineqs[:, 2, np.newaxis] + \
                  points[:, 2] * ineqs[:, 3, np.newaxis] + ineqs[:, 0, np.newaxis]

        inside = (np.sign(inside)+1).astype(bool)

        points_inside_all_planes = inside.all(axis=0)
        at_least_one_point_inside_all_planes = points_inside_all_planes.any()

        return at_least_one_point_inside_all_planes

    def my_construct(self,m):


        # iteratively slice convexes with planes
        self.split_count = 0
        self.face_count = 0

        self.get_bounding_box(m)
        plot_interface = True
        self.write_cells(m,self.bounding_poly)

        tree = Tree()
        tree.create_node(tag=self.split_count, identifier=self.split_count,data=self.bounding_poly) # root node

        for i,plane in enumerate(self.planes):

            children = list(tree.expand_tree(0,filter= lambda x: self.contains(x.data,self.points[i])))

            for child in children:


                if not tree[child].is_leaf():
                    continue
                current_node=tree[child].data

                hspace_positive, hspace_negative = [Polyhedron(ieqs=[inequality]) for inequality in
                                                    self._inequalities(plane)]

                hspace_positive = current_node.intersection(hspace_positive)
                hspace_negative = current_node.intersection(hspace_negative)

                # if hspace_positive.dim() != 3 or hspace_negative.dim() != 3:
                #     continue

                if(not hspace_positive.is_empty()): # should maybe adopt the dim != 3 method of the original code instead
                    self.split_count+=1
                    tree.create_node(tag=self.split_count,identifier=self.split_count,data=hspace_positive,parent=tree[child].identifier)
                    self.write_cells(m,hspace_positive)

                if(not hspace_negative.is_empty()):
                    self.split_count+=1
                    tree.create_node(tag=self.split_count,identifier=self.split_count,data=hspace_negative,parent=tree[child].identifier)
                    self.write_cells(m,hspace_negative,self.points[i])

                if plot_interface:
                    c = np.random.random(size=3)

                    if (not hspace_negative.is_empty()) and (not hspace_positive.is_empty()):
                        facet = hspace_positive.intersection(hspace_negative)
                        self.write_faces(m,facet)
                        self.face_count+=1

        a=4

    def sort_planes_by_absolute_occ(self,planes,points_tgt,occ_tgt):

        # check on which side of the plane the query points lie
        which_side = planes[:, 0, np.newaxis] * points_tgt[:, 0] \
                    + planes[:, 1, np.newaxis] * points_tgt[:, 1] \
                    + planes[:, 2, np.newaxis] * points_tgt[:, 2] \
                     + planes[:, 3, np.newaxis]

        which_side = (np.sign(which_side)+1).astype(bool)

        # check how many query points would get correctly split by the plane
        # we don't know the correct orientation of the plane, so we check for both possible orientations
        sortinga = (which_side == occ_tgt).sum(axis=1)
        sortingb = (np.invert(which_side) == occ_tgt).sum(axis=1)

        # high split value in either of the two orientations is good
        sorting = np.vstack((sortinga,sortingb))
        sorting = np.max(sorting,axis=0)

        # sort by values, high split values first
        sorting = np.flip(np.argsort(sorting))

        return sorting


    def sort_planes_by_surface_split(self, m, vertex_group, mode=Tree.DEPTH):

        ex = Exporter()



        self.get_bounding_box(m)
        cell_count = 0
        tree = Tree()
        dd = {"plane_ids": np.arange(vertex_group.planes.shape[0])}
        tree.create_node(tag=cell_count, identifier=cell_count, data=dd)  # root node

        children = tree.expand_tree(0, filter=lambda x: x.data["plane_ids"].shape[0], mode=mode)

        point_groups = []
        for pg in vertex_group.points_grouped:
            point_groups.append(pg.shape[0])

        mpg = max(point_groups)
        for i,pg in enumerate(vertex_group.points_grouped):
            pg = vertex_group.points_grouped[i]
            point_groups[i] = np.concatenate((pg,np.zeros(shape=(mpg-pg.shape[0],3))*np.nan),axis=0)

        point_groups = np.array(point_groups)

        plane_order = []

        for child in children:

            current_ids = tree[child].data["plane_ids"]

            ### the whole thing vectorized. doesn't really work for some reason
            #
            # planes = np.repeat(vertex_group.planes[np.newaxis,current_ids,:],current_ids.shape[0],axis=0)
            # pgs = np.repeat(point_groups[np.newaxis,current_ids,:,:],current_ids.shape[0],axis=0)
            #
            # which_side = planes[:,:,0,np.newaxis] * pgs[:,:,:,0] + planes[:,:,1,np.newaxis] * pgs[:,:,:,1] + planes[:,:,2,np.newaxis] * pgs[:,:,:,2] + planes[:,:,3,np.newaxis]
            #


            current_point_groups = point_groups[current_ids,:,:]
            left_right = []
            for id in current_ids:

                plane = vertex_group.planes[id,:]

                which_side = plane[0] * current_point_groups[:,:,0] + plane[1] * current_point_groups[:,:,1] + plane[2] * current_point_groups[:,:,2] + plane[3]

                which_side[np.isnan(which_side)] = 0

                # nans = np.isnan(which_side).sum(axis=-1)

                left = (which_side<=0).all(axis=-1).sum()
                right= (which_side>=0).all(axis=-1).sum()

                left_right.append([left,right])


            left_right = np.array(left_right)
            left_right = left_right.sum(axis=1)
            best_plane = np.argmax(left_right)
            plane_order.append(current_ids[best_plane])
            plane = vertex_group.planes[current_ids[best_plane]]
            which_side = plane[0] * current_point_groups[:, :, 0] + plane[1] * current_point_groups[:, :, 1] + plane[2] * current_point_groups[:, :, 2] + plane[3]
            which_side[np.isnan(which_side)] = 0
            which_side[best_plane,:] = np.nan

            ### problem is that here I miss the planes that are split by the best_plane, ie that are not fully left or right
            ### what I need TODO is to split these point groups of these planes and also put them to left and right accordingly
            ### it will just get a bit complex with the indexing and I have to recalculate self.bounds, but should be doable
            ### what should probably be done is to set a certain threshold for a plane to be part of a side; ie if only eg 5
            ### points are part of a side then just drop them as inliers to the plane to save time
            ### so instead of (which_side<=0).all(axis=-1) do something like (which_side<=0).sum(axis=-1)/n_inliers > th

            left_ids = current_ids[(which_side<=0).all(axis=-1)]
            dd = {"plane_ids": left_ids}
            cell_count = cell_count+1
            tree.create_node(tag=cell_count, identifier=cell_count, data=dd, parent=tree[child].identifier)

            right_ids = current_ids[(which_side>=0).all(axis=-1)]
            dd = {"plane_ids": right_ids}
            cell_count = cell_count+1
            tree.create_node(tag=cell_count, identifier=cell_count, data=dd, parent=tree[child].identifier)

            a=5


        self.planes = self.planes[plane_order]
        self.bounds = self.bounds[plane_order]
        self.points = self.points[plane_order]

        return 0






















    def sort_planes_by_occ_ratio(self, m, planes, points_tgt, occ_tgt, count):

        # check on which side of the plane the query points lie
        which_side = planes[:, 0, np.newaxis] * points_tgt[:, 0] \
                    + planes[:, 1, np.newaxis] * points_tgt[:, 1] \
                    + planes[:, 2, np.newaxis] * points_tgt[:, 2] \
                     + planes[:, 3, np.newaxis]

        which_side = (np.sign(which_side)+1).astype(bool)

        # check how many query points would get correctly split by the plane
        # we don't know the correct orientation of the plane, so we check for both possible orientations
        lefta = np.stack((which_side, (np.repeat(occ_tgt[np.newaxis, :], which_side.shape[0], axis=0))))
        lefta = lefta.all(axis=0)
        lefta = lefta.sum(axis=1)
        leftaf = lefta/which_side.sum(axis=1)

        righta = np.stack((np.invert(which_side), np.invert(np.repeat(occ_tgt[np.newaxis, :], which_side.shape[0], axis=0))))
        righta = righta.all(axis=0)
        rightaf = righta.sum(axis=1)/np.invert(which_side).sum(axis=1)

        orientationa = np.vstack((leftaf, rightaf))
        best_sidea, best_planea = np.unravel_index(np.nanargmax(orientationa), shape=orientationa.shape)
        orientationa = orientationa[best_sidea,best_planea]

        epath = os.path.join(os.path.dirname(m["abspy"]["partition"]))
        if orientationa > 0.99:

            if best_sidea == 0:
                self.exporter.export_deleted_points(epath,points_tgt[which_side[best_planea,:],:],count)
                points_tgt = np.delete(points_tgt,which_side[best_planea,:],axis=0)
                occ_tgt = np.delete(occ_tgt,which_side[best_planea,:],axis=0)
            else:
                self.exporter.export_deleted_points(epath,points_tgt[np.invert(which_side)[best_planea,:],:],count)
                points_tgt = np.delete(points_tgt,np.invert(which_side)[best_planea,:],axis=0)
                occ_tgt = np.delete(occ_tgt,np.invert(which_side)[best_planea,:],axis=0)

            return best_planea, points_tgt, occ_tgt



        which_side = np.invert(which_side)

        leftb = np.stack((which_side, (np.repeat(occ_tgt[np.newaxis, :], which_side.shape[0], axis=0))))
        leftb = leftb.all(axis=0)
        leftbf = leftb.sum(axis=1)/which_side.sum(axis=1)

        rightb = np.stack((np.invert(which_side), np.invert(np.repeat(occ_tgt[np.newaxis, :], which_side.shape[0], axis=0))))
        rightb = rightb.all(axis=0)
        rightbf = rightb.sum(axis=1)/np.invert(which_side).sum(axis=1)

        orientationb = np.vstack((leftbf,rightbf))
        best_sideb, best_planeb = np.unravel_index(np.nanargmax(orientationb), shape=orientationb.shape)
        orientationb = orientationb[best_sideb,best_planeb]

        if orientationb > 0.99:

            if best_sideb == 0:
                self.exporter.export_deleted_points(epath,points_tgt[which_side[best_planeb, :], :],count)
                points_tgt = np.delete(points_tgt, which_side[best_planeb, :], axis=0)
                occ_tgt = np.delete(occ_tgt, which_side[best_planeb, :], axis=0)
            else:
                self.exporter.export_deleted_points(epath,points_tgt[np.invert(which_side)[best_planeb, :], :],count)
                points_tgt = np.delete(points_tgt, np.invert(which_side)[best_planeb, :], axis=0)
                occ_tgt = np.delete(occ_tgt, np.invert(which_side)[best_planeb, :], axis=0)

            return best_planeb, points_tgt, occ_tgt

        if orientationa >= orientationb:

            return best_planea, points_tgt, occ_tgt

        else:

            return best_planeb, points_tgt, occ_tgt



    def my_prioritise_planes(self,m):

        logger.info('my prioritising planar primitives')

        occs = np.load(m["occ"])
        points_tgt = occs['points']
        occ_tgt = np.unpackbits(occs['occupancies']).astype(bool)


        # sorting = self.sort_planes_by_absolute_occ(planes=self.planes,points_tgt=points_tgt,occ_tgt=occ_tgt)
        # sorting = self.sort_planes_by_occ_ratio(m,planes=self.planes, points_tgt=points_tgt, occ_tgt=occ_tgt)

        planes = deepcopy(self.planes)
        sorting = []

        count=0
        for _ in planes:
            best_plane_idx,points_tgt,occ_tgt = self.sort_planes_by_occ_ratio(m, planes=planes, points_tgt=points_tgt, occ_tgt=occ_tgt,count=count)
            planes[best_plane_idx,:] = None

            sorting.append(best_plane_idx)
            count+=1




        # reorder both the planes and their bounds
        self.planes = self.planes[sorting]
        self.bounds = self.bounds[sorting]
        self.points = self.points[sorting]

        self.exporter.export_planes(os.path.join(os.path.dirname(m["planes"])),self.planes,self.points)


        a=5

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

    # def _my_pad_bound(self,bounds, padding=0.00):
    #
    #     """
    #     pad each dimension separately as in Kinetic_Propagation::pre_init_main_bounding_box()
    #     as opposed to the _pad_bounds function above that pads with the max of all dimensions
    #     :param bounds:
    #     :param padding:
    #     :return:
    #     """
    #
    #     a=5


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

    def visualise(self, indices_cells=None, temp_dir='./'):
        """
        Visualise the cells using trimesh.

        pyglet installation is needed for the visualisation.

        Parameters
        ----------
        indices_cells: None or (n,) int
            Indices of cells to be visualised
        temp_dir: str
            Temp dir to save intermediate visualisation
        """
        if self.constructed:
            try:
                import pyglet
            except ImportError:
                logger.warning('pyglet installation not found; skip visualisation')
                return
            if indices_cells is not None and len(indices_cells) == 0:
                raise ValueError('no indices provided')

            filename_stem = ''.join(choices(string.ascii_uppercase + string.digits, k=5))
            filename_mesh = filename_stem + '.obj'
            filename_mtl = filename_stem + '.mtl'

            self.save_obj(filepath=temp_dir + filename_mesh, indices_cells=indices_cells, use_mtl=True)
            scene = trimesh.load_mesh(temp_dir + filename_mesh)
            scene.show()
            os.remove(temp_dir + filename_mesh)
            os.remove(temp_dir + filename_mtl)
        else:
            raise RuntimeError('cell complex has not been constructed')

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
        if self.constructed:
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
        else:
            raise RuntimeError('cell complex has not been constructed')

    def save_plm(self, filepath, indices_cells=None):
        """
        Save polygon soup of indexed convexes to a plm file (polyhedron mesh in Mapple).

        Parameters
        ----------
        filepath: str or Path
            Filepath to save plm file
        indices_cells: (n,) int
            Indices of cells to save to file
        """
        if self.constructed:
            # create the dir if not exists
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            num_vertices = 0
            info_vertices = ''
            info_facets = ''
            info_header = ''

            cells = [self.cells[i] for i in indices_cells] if indices_cells is not None else self.cells

            scene = None
            for cell in cells:
                scene += cell.render_solid()
                num_vertices += cell.n_vertices()

            info_header += '#vertices {}\n'.format(num_vertices)
            info_header += '#cells {}\n'.format(len(cells))

            with open(filepath, 'w') as f:
                contents = scene.obj_repr(scene.default_render_params())
                for o in range(len(cells)):
                    info_vertices += '\n'.join([st[2:] for st in contents[o][2]]) + '\n'
                    info_facets += str(len(contents[o][3])) + '\n'
                    for st in contents[o][3]:
                        info_facets += str(len(st[2:].split())) + ' '  # number of vertices on this facet
                        info_facets += ' '.join([str(int(n) - 1) for n in st[2:].split()]) + '\n'
                f.writelines(info_header + info_vertices + info_facets)

        else:
            raise RuntimeError('cell complex has not been constructed')
