import logging
import os, sys
from pathlib import Path
import numpy as np
from sage.all import polytopes, QQ, Polyhedron
from pyplane.pyplane import PyPlane, SagePlane, ProjectedConvexHull
from pyplane.export import PlaneExporter
import copy
from .logger import make_logger

class VertexGroup:
    """
    Class for manipulating planar primitives.
    """

    def __init__(self, input_file, prioritise = None,
                 points_type="inliers", total_sample_count=100000, recolor=False, logging_level=logging.ERROR):
        """
        Init VertexGroup.
        Class for manipulating planar primitives.

        Parameters
        ----------
        filepath: str or Path
            Filepath to vertex group file (.vg) or binary vertex group file (.bvg)
        """


        # set random seed to have deterministic results for point sampling and filling of convex hull point arrays.
        np.random.seed(42)

        self.logger = make_logger(name="COMPOD",level=logging_level)

        self.input_file = input_file
        self.prioritise = prioritise
        self.total_sample_count = total_sample_count
        self.points_type = points_type
        self.recolor = recolor

        ending = os.path.splitext(self.input_file)[1]
        if ending == ".npz":
            self._process_npz()
        else:
            self.logger.error("{} is not a valid file type for planes. Only .npz files are allowed.".format(ending))
            sys.exit(1)



    def _recolor_planes(self):

        from fancycolor.color import FancyColor

        bbox = np.vstack((self.points.min(axis=0),self.points.max(axis=0)))

        fc=FancyColor(bbox)
        cols = []
        for p in self.polygons:
            pt = copy.deepcopy(p.centroid)
            cols.append(fc.get_rgb_from_xyz(pt))
        self.plane_colors = np.array(cols)


    def _prioritise_planes(self, mode):
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

        def _vertical_planes(slope_threshold=0.9, epsilon=10e-5):
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


        self.logger.debug('Prioritise planar primitive with mode {}'.format(mode))

        indices_sorted_planes = np.arange(len(self.planes))

        if mode == "random":
            np.random.shuffle(indices_sorted_planes)
            return indices_sorted_planes
        elif mode == "vertical":
            indices_vertical_planes = _vertical_planes(slope_threshold=0.9)
            bool_vertical_planes = np.in1d(indices_sorted_planes, indices_vertical_planes)
            return np.append(indices_sorted_planes[bool_vertical_planes],
                                         indices_sorted_planes[np.invert(bool_vertical_planes)])
        elif mode == "n-points":
            npoints = []
            for pg in self.points_grouped:
                npoints.append(pg.shape[0])
            npoints = np.array(npoints)
            return np.argsort(npoints)[::-1]
        elif mode == 'volume':
            volume = np.prod(self.bounds[:, 1, :] - self.bounds[:, 0, :], axis=1)
            return np.argsort(volume)[::-1]
        elif mode == 'norm':
            sizes = np.linalg.norm(self.bounds[:, 1, :] - self.bounds[:, 0, :], ord=2, axis=1)
            return np.argsort(sizes)[::-1]
        elif mode == 'area':
            return np.argsort(self.polygon_areas)[::-1]
        elif mode == "product" or mode == "product-earlystop" or mode == "sum" or mode == "sum-earlystop":
            return indices_sorted_planes
        else:
            raise NotImplementedError




    def _fill_hull_vertices(self):

        hull_vertices = []

        for i,v in enumerate(self.hull_vertices):

            # hp = np.array(self.convex_hulls[i].hull_points)

            fill_vertices = np.random.choice(self.groups[i],self.n_fill-v.shape[0])
            fhv = np.concatenate((v,fill_vertices))
            hull_vertices.append(fhv)

        self.hull_vertices = np.array(hull_vertices)

    def _process_npz(self):
        """
        Start processing vertex group.
        """

        data = np.load(self.input_file)

        # read the data and make the point groups
        self.planes = data["group_parameters"].astype(np.float32)
        self.points = data["points"].astype(np.float32)
        self.normals = data["normals"].astype(np.float32)
        npoints = data["group_num_points"].flatten()
        verts = data["group_points"].flatten()
        self.plane_colors = data["group_colors"]

        self.halfspaces = []
        self.polygons = []
        self.polygon_areas = []
        self.projected_points = np.zeros(shape=(self.points.shape[0],2))
        self.groups = []
        self.hull_vertices = []
        self.n_fill = 0
        last = 0
        for i,npp in enumerate(npoints):
            ## make the point groups
            vert_group = verts[last:(npp+last)]
            assert vert_group.dtype == np.int32
            pts = self.points[vert_group]
            self.groups.append(vert_group)

            # TODO: i am computing the convex hull twice below; not necessary

            ## make the polys
            ## make a trimesh of each input polygon, used for getting the area of each input poly for area based sorting
            pl = PyPlane(self.planes[i])
            poly = pl.get_trimesh_of_projected_points(pts,type="convex_hull")
            self.polygons.append(poly)
            self.polygon_areas.append(poly.area)

            ## this is used for finding points associated to facets of the partition for normal based occupancy voting
            self.projected_points[vert_group] = pl.project_points_to_plane(pts)[:,:2]

            ## this is used for _get_best_plane function
            pch = ProjectedConvexHull(self.planes[i],pts)
            # self.projected_points[vert_group] = pch.all_projected_points_2d
            self.hull_vertices.append(vert_group[pch.hull.vertices])
            n_hull_vertices = len(pch.hull.vertices)
            if n_hull_vertices > self.n_fill:
                self.n_fill = n_hull_vertices

            self.halfspaces.append([Polyhedron(ieqs=[inequality]) for inequality in self._inequalities(self.planes[i])])

            last += npp

        assert self.points.shape[0] == self.projected_points.shape[0]

        if self.recolor:
            self._recolor_planes()
            # save with new colors
            data = dict(data)
            data["group_colors"] = self.plane_colors
            np.savez(self.input_file,**data)

        self.polygons = np.array(self.polygons)
        self.polygon_areas = np.array(self.polygon_areas)


        if self.points_type == "samples":
            self.logger.info("Sample polygons with {} points".format(self.total_sample_count))

            ### scale sample_count_per_area by total area of input polygons. like this n_sample_points should roughly be constant for each mesh + (convex hull points)
            self.sample_count_per_area = self.total_sample_count / self.polygon_areas.sum()

            self.points = []
            self.normals = None
            self.groups = []
            self.hull_vertices = []
            n_points = 0
            for i, poly in enumerate(self.polygons):
                n = 3 + int(self.sample_count_per_area * self.polygon_areas[i])
                sampled_points = poly.sample(n)
                sampled_points = np.concatenate((poly.vertices, sampled_points), axis=0, dtype=np.float32)
                self.points.append(sampled_points)
                self.groups.append(np.arange(len(sampled_points))+n_points)
                self.hull_vertices.append(np.arange(len(poly.vertices))+n_points)
                n_points+=sampled_points.shape[0]

            self.points = np.concatenate(self.points)

        elif self.points_type == "inliers":
            self.logger.info("Use {} inlier points of polygons".format(np.concatenate(self.groups).shape[0]))
        else:
            print("{} is not a valid point_type. Only 'inliers' or 'samples' are allowed.".format(self.points_type))
            raise NotImplementedError


        # fill the hull_vertices array to make it a matrix instead of jagged array for an efficient _get_best_plane function with matrix multiplications
        # if torch.nested.nested_tensor ever supports broadcasting and dot products, the code could be simplified a lot.
        self.n_fill = self.n_fill*2
        self._fill_hull_vertices()


        ## export planes and samples
        pe = PlaneExporter()
        pt_file = os.path.splitext(self.input_file)[0]+"_samples.ply"
        plane_file =  os.path.splitext(self.input_file)[0]+'.ply'
        pe.save_points_and_planes([pt_file,plane_file],points=self.points, normals=self.normals, groups=self.groups, planes=self.planes, colors=self.plane_colors)

        if self.prioritise is not None:
            order = self._prioritise_planes(self.prioritise)
            self.plane_order = order
            self.planes = self.planes[order]
            self.halfspaces = list(np.array(self.halfspaces)[order])
            self.groups = list(np.array(self.groups,dtype=object)[order])
            self.hull_vertices = self.hull_vertices[order]
            self.plane_colors = self.plane_colors[order]
        else:
            self.plane_order = np.arange(len(self.planes))
            self.logger.info("No plane prioritisation applied")

        del self.polygons
        del self.polygon_areas

        self.bounds = []
        for group in self.groups:
            pts = self.points[group]
            bounds = np.array([np.amin(pts, axis=0), np.amax(pts, axis=0)])
            self.bounds.append(bounds)
        self.bounds = np.array(self.bounds)



    def _inequalities(self,plane):
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

