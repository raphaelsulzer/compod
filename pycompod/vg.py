import logging, os, sys, trimesh
import math

import numpy as np
from collections import defaultdict
from sage.all import polytopes, QQ, Polyhedron
from sklearn.neighbors import NearestNeighbors

from .logger import make_logger
from .plane import PyPlane, ProjectedConvexHull
from .export_plane import PlaneExporter

class VertexGroup:
    """
    Class for manipulating planar primitives.
    """

    def __init__(self, input_file, prioritise="area", merge_duplicate_planes=True, epsilon=None, alpha=0.05,
                 points_type="inliers", total_sample_count=100000, debug_export=False,
                 recolor=False, verbosity=logging.WARN):
        """
        Init VertexGroup.
        Class for manipulating planar primitives.

        Parameters
        ----------
        input_file: str to .npz file
        prioritise: Sort planes by this value.
        merge_duplicate_planes: Merge vertex groups with same plane parameters
        epsilon: Angle for clustering planes
        alpha: Distance for clustering planes
        points_type: Whether to use the provided inliers, or sample new points on the 2D projected convex hull of inliers.
        total_sample_count: how many points to sample. Only applicable if points_type="sampling".
        debug_export: Whether to export debug files.
        recolor: Recolor planes and save back new coloring to .npz files.
        verbosity: Verbosity as python logging_level.
        """

        # set random seed to have deterministic results for point sampling and filling of convex hull point arrays.
        np.random.seed(42)

        self.logger = make_logger(name="COMPOD",level=verbosity)

        self.input_file = input_file
        self.merge_duplicate_planes = merge_duplicate_planes
        self.prioritise = prioritise
        self.epsilon = epsilon
        self.alpha = alpha
        self.total_sample_count = total_sample_count
        self.points_type = points_type
        self.recolor = recolor
        self.debug_export = debug_export

        self.plane_exporter = PlaneExporter(verbosity=verbosity)

        ending = os.path.splitext(self.input_file)[1]
        if ending == ".npz":
            self._read_npz()
        elif ending == ".vg":
            self._read_vg()
        else:
            self.logger.error("{} is not a valid file type for planes. Only .npz or .vg files are allowed.".format(ending))
            return 1

        if self.epsilon is not None:
            self._cluster_planes_with_abc_and_origin_dist()

        self._prepare_for_polyhedral_complex_construction()

    def _read_npz(self):
        """
        Load vertex groups and planes from npz file.
        :return:
        """
        data = np.load(self.input_file)

        type = np.float64
        # read the data and make the point groups
        self.points = data["points"].astype(type)
        self.normals = data["normals"].astype(type)
        self.planes = data["group_parameters"].astype(type)
        self.plane_colors = data["group_colors"]
        # self.classes = data.get("classes", np.ones(len(self.points),dtype=np.int32))
        self.classes = data.get("classes", None)
        if not len(self.classes):
            self.classes = None
        self.groups = self._load_point_groups(data["group_points"].flatten(), data["group_num_points"].flatten())

        self.logger.info(
            "Loaded {} inlier points of {} planes".format(np.concatenate(self.groups).shape[0], len(self.planes)))

        ## recolor planes and support points and save new coloring to .npz
        if self.recolor:
            self._recolor_planes()
            # save with new colors
            data = dict(data)
            data["group_colors"] = self.plane_colors
            np.savez(self.input_file,**data)

    def _load_point_groups(self,point_ids,sizes):
        current = 0
        groups = []
        for n in sizes:
            pids = point_ids[current:(n+current)]
            assert pids.dtype == np.int32
            groups.append(pids)
            current+=n
        return groups

    def _read_vg(self):
        self.points = []
        self.normals = []
        self.groups = []
        self.planes = []
        self.plane_colors = []
        self.classes = None

        with open(self.input_file, 'r') as file:
            # Extract numbers from the current line
            number_line = file.readline()
            line_numbers = self.extract_numbers_from_line(number_line)
            n_poins = line_numbers[0]
            # read the n_points
            for i in range(n_poins):
                line = file.readline()
                [x, y, z] = line.split()
                self.points.append([float(x), float(y), float(z)])

            number_line = file.readline()

            for i in range(n_poins):
                line = file.readline()

            number_line = file.readline()

            for i in range(n_poins):
                line = file.readline()
                [x, y, z] = line.split()
                self.normals.append([float(x), float(y), float(z)])

            number_line = file.readline()
            line_numbers = self.extract_numbers_from_line(number_line)
            n_groups = line_numbers[0]

            for i in range(n_groups):
                number_line = file.readline() # group_type
                number_line = file.readline() # num_group_parameters: 4
                number_line = file.readline() # group_parameters: a b c d
                [t, a, b, c, d] = number_line.split()
                h_list = [float(a), float(b), float(c), float(d)]
                self.planes.append(h_list)
                number_line = file.readline() # group_label: unknown
                number_line = file.readline() # group_color: r g b
                [t, r, g, b] = number_line.split()
                color_list = [float(r), float(g), float(b)]
                self.plane_colors.append(color_list)

                number_line = file.readline() # roup_num_point
                line_numbers = self.extract_numbers_from_line(number_line)
                line = file.readline()
                str_numbers = line.split()
                int_list = [int(num) for num in str_numbers]
                self.groups.append(np.array(int_list,dtype=np.int32))
                number_line = file.readline()

        type = np.float64
        self.points = np.array(self.points,dtype=type)
        self.normals = np.array(self.normals,dtype=type)
        self.planes = np.array(self.planes,dtype=type)
        self.plane_colors = np.array(self.plane_colors,dtype=np.int32)

    def _recolor_planes(self):

        from fancycolor import GradientColor3D

        bbox = np.vstack((self.points.min(axis=0),self.points.max(axis=0)))

        fc=GradientColor3D(bbox)
        cols = []
        for group in self.groups:
            pt = self.points[group].mean(axis=0)
            # pt = copy.deepcopy(p.centroid)
            cols.append(fc.get_rgb_from_xyz(pt))
        self.plane_colors = np.array(cols)


    def _prioritise_planes(self, mode):
        """
        Prioritise certain planes. Mainly taken from abspy.

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
            for pg in self.groups:
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
        elif mode == 'class':
            # {1: "Unclassified", 2: "Ground", 3: "Low_Vegetation", 4: "Medium_Vegetation", 5: "High_Vegetation",
            #  6: "Building", 9: "Water",
            #  17: "17", 64: "64", 65: "65", 66: "Floor", 67: "Walls"}

            class_groups = defaultdict(list)
            # get majority class for each plane
            for i,pg in enumerate(self.groups):
                if np.isclose(self.planes[i,2],0,atol=0.05):
                    class_groups["Walls"].append(i)
                    continue

                unique, count = np.unique(self.classes[pg],return_counts=True)
                max_class = unique[np.argmax(count)]
                if max_class == 6:
                    class_groups["Building"].append(i)
                elif max_class == 66:
                    class_groups["Floor"].append(i)
                elif max_class == 67:
                    class_groups["Walls"].append(i)
                elif max_class == 68:
                    class_groups["Etage"].append(i)
                else:
                    class_groups["Misc"].append(i)

            order = []
            for cl in ["Floor","Etage","Walls","Building","Misc"]:
                group_ids = np.array(class_groups[cl]).astype(int)
                polys = self.polygon_areas[group_ids]
                ord = np.argsort(polys)[::-1]
                order.append(group_ids[ord])

            # TODO: this doesn't actually work, because my algo resorts the planes. what I need to do is first insert
            #  the floor and walls. and then insert the roof

            return np.concatenate(order)


        elif mode == "product" or mode == "product-earlystop" or mode == "sum" or mode == "sum-earlystop":
            return indices_sorted_planes
        else:
            raise NotImplementedError


    def _fill_hull_vertices(self):

        hull_vertices = []
        for i,v in enumerate(self.hull_vertices):
            try:
                fill_vertices = np.random.choice(self.groups[i],self.n_fill-v.shape[0])
            except:
                a=4
            fhv = np.concatenate((v,fill_vertices))
            hull_vertices.append(fhv)
        self.hull_vertices = np.array(hull_vertices)

    def _merge_duplicate_planes(self):

        n_input = len(self.planes)

        # TODO: there is a cleaner way to do this, by using: import partial; defaultdict(partial(numpy.ndarray, 0)) and
        #  then I don't need the if statement anymore, and can just
        # use np.append(groups[inverse[i]],self.groups[i])

        groups = defaultdict(int)
        primitive_ids = defaultdict(list)
        plane_colors = defaultdict(list)
        halfspaces = defaultdict(list)
        hull_vertices = defaultdict(list)
        
        unique, inverse = np.unique(self.planes, return_inverse=True, axis=0)
        for i in range(len(self.planes)):
            if isinstance(groups[inverse[i]],int): ## hacky way to check if this item already has a value or is empty, ie has the default int assigned
                groups[inverse[i]] = self.groups[i]
                hull_vertices[inverse[i]] = self.hull_vertices[i]
            else:
                groups[inverse[i]] = np.concatenate((groups[inverse[i]], self.groups[i]))
                hull_vertices[inverse[i]] = np.concatenate((hull_vertices[inverse[i]], self.hull_vertices[i]))
                if len(hull_vertices[inverse[i]]) > self.n_fill:
                    self.n_fill = len(hull_vertices[inverse[i]])

            plane_colors[inverse[i]] = self.plane_colors[i]
            halfspaces[inverse[i]] = self.halfspaces[i]
            primitive_ids[inverse[i]]+=[i]

        self.groups = list(groups.values())
        self.planes = unique[list(groups.keys())]
        self.plane_colors = np.array(list(plane_colors.values()))
        self.halfspaces = list(halfspaces.values())
        self.hull_vertices = list(hull_vertices.values())

        self.logger.info("Merged duplicate planes from {} to {}".format(n_input, self.planes.shape[0]))
        self.logger.debug("Ie also merged point groups from the same plane to one point group and so on.")
        
        # put in the id of the merged primitive, ie also the plane, and get out the 1 to n input primitives that were merged for it
        return list(primitive_ids.values())


    def _cluster_planes_with_abc_and_projection_dist(self):
        """Project points from plane a to plane b and vice versa. Problem is that for large models,
        such as chicago, very similar planes in angle will still not cluster."""

        def acos(x):
            # range is -1 to 1 !!
            return (-0.69813170079773212 * x * x - 0.87266462599716477) * x + 1.5707963267948966

        self.logger.info("Cluster coplanar planes with epsilon = {} and alpha = {}".format(self.epsilon,self.alpha))

        ### orient planes to a corner, important to run this before cluster_planes() so that planes with the same normal but d = -d are not clustered
        self._orient_planes(to_corner=True)

        epsilon_cosine = self.alpha
        epsilon_euclidean = self.epsilon

        # def custom_distance(x, y, epsilon_cosine, epsilon_euclidean):
        def custom_distance(x, y):

            dist1 = np.abs(np.dot(x[4:],y[:3])+y[3])
            dist2 = np.abs(np.dot(y[4:],x[:3])+x[3])
            euclidean_dist = (dist1+dist2)/2

            cosine_dist = np.dot(x[:3],y[:3])
            # better to keep the degree scaling in here, even if it is more expensive, but it is awful to tune the epsilon alpha parameter otherwise
            scaled_cosine_dist = (acos(cosine_dist)*180/math.pi) / epsilon_cosine
            scaled_euclidean_dist = euclidean_dist / epsilon_euclidean
            return scaled_cosine_dist+scaled_euclidean_dist

        self.plane_centroids = []
        for gr in self.groups:
            pts = self.points[gr]
            self.plane_centroids.append(pts.mean(axis=0))
        self.plane_centroids = np.array(self.plane_centroids)
        self.features = np.hstack((self.planes,self.plane_centroids))

        ### clustering

        clusters = NearestNeighbors(n_jobs=-1, radius=0.99, metric=custom_distance).fit(self.features)
        clusters = clusters.radius_neighbors(return_distance=False)
        
        processed = np.zeros(self.planes.shape[0])
        new_planes = np.zeros(self.planes.shape)
        for i,cl in enumerate(clusters):
            if processed[i]:
                continue
            if not len(cl):
                new_planes[i] = self.planes[i]
                processed[i] = True
            else:
                ids = np.hstack((cl, i))
                new_planes[ids] = np.mean(self.planes[ids,:],axis=0)
                processed[ids] = True
            
        
        self.planes = new_planes

    def _cluster_planes_with_abc_and_origin_dist(self):

        def acos(x):
            # range is -1 to 1 !!
            return (-0.69813170079773212 * x * x - 0.87266462599716477) * x + 1.5707963267948966

        self.logger.info("Cluster coplanar planes with epsilon = {} and alpha = {}".format(self.epsilon, self.alpha))

        ### orient planes to a corner, important to run this before cluster_planes() so that planes with the same normal but d = -d are not clustered
        self._orient_planes(to_corner=True)

        epsilon_cosine = self.alpha
        epsilon_euclidean = self.epsilon

        # def custom_distance(x, y, epsilon_cosine, epsilon_euclidean):
        def custom_distance(x, y):

            euclidean_dist = np.abs(x[3] - y[3])
            cosine_dist = np.dot(x[:3], y[:3])
            # better to keep the degree scaling in here, even if it is more expensive, but it is awful to tune the epsilon alpha parameter otherwise
            scaled_cosine_dist = (acos(cosine_dist) * 180 / math.pi) / epsilon_cosine
            scaled_euclidean_dist = euclidean_dist / epsilon_euclidean
            return scaled_cosine_dist + scaled_euclidean_dist

        ### clustering
        clusters = NearestNeighbors(n_jobs=-1, radius=0.99, metric=custom_distance).fit(self.planes)
        clusters = clusters.radius_neighbors(return_distance=False)

        processed = np.zeros(self.planes.shape[0])
        new_planes = np.zeros(self.planes.shape)
        for i, cl in enumerate(clusters):
            if processed[i]:
                continue
            if not len(cl):
                new_planes[i] = self.planes[i]
                processed[i] = True
            else:
                ids = np.hstack((cl, i))
                new_planes[ids] = np.mean(self.planes[ids, :], axis=0)
                processed[ids] = True

        self.planes = new_planes


    def _orient_planes(self,to_corner=False):

        if to_corner:
            corner = self.points.max(axis=0)
            for i, pl in enumerate(self.planes):
                if np.dot(pl[:3], corner) < 0:
                    self.planes[i] = -pl
        else:
            for i,pl in enumerate(self.planes):

                point_normals = self.normals[self.groups[i]]
                plane_normal = np.mean(point_normals,axis=0)
                if np.dot(pl[:3],plane_normal) < 0:
                    self.planes[i] = -pl


    def extract_numbers_from_line(self, line):
        import re
        # Define a regular expression pattern to find all numbers in the line
        pattern = r'\d+\.?\d*'  # This will match integers and decimal numbers

        # Find all matches of the pattern in the line
        matches = re.findall(pattern, line)

        # Convert matches to float or int
        numbers = [float(match) if '.' in match else int(match) for match in matches]

        return numbers


    def _prepare_for_polyhedral_complex_construction(self):
        """Create halfspaces, and polygons from planes, necessary for the PolyhedralComplex() class.
        Optionally, resamples planar primitives, instead of using provided inliers.
        Optionally, merge dublicate planes."""


        ### orient planes according to the mean normal orientation of it's input points
        ### cannot actually use this before merging, because I want to merge oppositely oriented planes!
        ### can also not do it after merge, see below!
        # self._orient_planes()

        self.halfspaces = []
        self.polygons = []
        self.polygon_areas = []
        self.projected_points = np.zeros(shape=(self.points.shape[0],2))
        self.hull_vertices = []
        self.n_fill = 0
        for i, vert_group in enumerate(self.groups):
            self.halfspaces.append([Polyhedron(ieqs=[inequality]) for inequality in self._inequalities(self.planes[i])])
            pts = self.points[vert_group]

            # TODO: i am computing the convex hull twice below; not necessary
            ## this is only used for sorting the polys by area
            pl = PyPlane(self.planes[i])
            try:
                poly = pl.get_trimesh_of_projected_points(pts, type="convex_hull")
                self.polygons.append(poly)
                self.polygon_areas.append(poly.area)
            except:
                self.logger.warning("Degenerate input polygon.")
                self.polygons.append(trimesh.Trimesh)
                self.polygon_areas.append(0)

            ## this is used for finding points associated to facets of the partition for normal based occupancy voting
            self.projected_points[vert_group] = pl.to_2d(pts)

            ## this is used for _get_best_plane function
            try:
                pch = ProjectedConvexHull(self.planes[i], pts)
                self.hull_vertices.append(vert_group[pch.hull.vertices])
                n_hull_vertices = len(pch.hull.vertices)
            except:
                self.hull_vertices.append(vert_group)
                n_hull_vertices = len(vert_group)

            if n_hull_vertices > self.n_fill:
                self.n_fill = n_hull_vertices

        assert self.points.shape[0] == self.projected_points.shape[0]
        self.polygons = np.array(self.polygons)
        self.polygon_areas = np.array(self.polygon_areas)


        if self.points_type == "samples":
            self.logger.info("Sample a total of {} points on {} polygons".format(self.total_sample_count,len(self.polygons)))
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
                sampled_points = np.concatenate((poly.vertices, sampled_points), axis=0, dtype=np.float64)
                self.points.append(sampled_points)
                self.groups.append(np.arange(len(sampled_points),dtype=np.int32)+n_points)
                self.hull_vertices.append(np.arange(len(poly.vertices),dtype=np.int32)+n_points)
                n_points+=sampled_points.shape[0]
            self.points = np.concatenate(self.points)
        elif self.points_type == "inliers":
            pass
        else:
            print("{} is not a valid point_type. Only 'inliers' or 'samples' are allowed.".format(self.points_type))
            raise NotImplementedError

        if self.debug_export:
            ## export planes and samples
            pt_file = os.path.splitext(self.input_file)[0]+"_samples.ply"
            plane_file =  os.path.splitext(self.input_file)[0]+'.ply'
            self.plane_exporter.save_points_and_planes(point_filename=pt_file,plane_filename=plane_file,points=self.points, normals=self.normals, groups=self.groups, planes=self.planes, colors=self.plane_colors)

        if self.prioritise is not None:
            order = self._prioritise_planes(self.prioritise)
            self.plane_order = order
            self.groups = np.array(self.groups,dtype=object)[order]
            if not self.groups[0].dtype == np.int32: # this line is only here for the rare case that all group arrays are of the same size, then numpy sets the dtype of each individual array to object
                self.groups = list(self.groups.astype(np.int32))
            else:
                self.groups = list(self.groups)
            self.planes = self.planes[order]
            self.plane_colors = self.plane_colors[order]
            self.halfspaces = list(np.array(self.halfspaces)[order])
            self.hull_vertices = np.array(self.hull_vertices,dtype=object)[order]
            if not self.hull_vertices[0].dtype == np.int32: # this line is only here for the rare case that all hull_vertices arrays are of the same size, then numpy sets the dtype of each individual array to object
                self.hull_vertices = self.hull_vertices.astype(np.int32)
        else:
            self.plane_order = np.arange(len(self.planes))
            self.logger.info("No plane prioritisation applied")
        del self.polygons
        del self.polygon_areas

        ## Note that we keep the prioritisation of planes when merging duplicates
        if self.merge_duplicate_planes:
            self.merged_plane_from_input_planes = self._merge_duplicate_planes()
        ## even if duplicate planes are not merged, check for them, for evaluation purposes
        else:
            self.merged_plane_from_input_planes = []
            for p in self.planes:
                self.merged_plane_from_input_planes.append(np.where((p==self.planes).all(axis=1))[0])

        ### orient planes according to the mean normal orientation of it's input points
        ### leads to a crash to do this down here, most likely because hspace orientations do not correspond to planes anymore
        # self._orient_planes()

        ## export planes and samples
        if self.debug_export:
            pt_file = os.path.splitext(self.input_file)[0]+"_samples_merged.ply"
            plane_file =  os.path.splitext(self.input_file)[0]+'_merged.ply'
            self.plane_exporter.save_points_and_planes(plane_filename=plane_file,point_filename=pt_file,points=self.points, normals=self.normals, groups=self.groups, planes=self.planes, colors=self.plane_colors)

        ## fill the hull_vertices array to make it a matrix instead of jagged array for an efficient _get_best_plane function with matrix multiplications
        ## if torch.nested.nested_tensor ever supports broadcasting and dot products, the code could be simplified a lot.
        self.n_fill = self.n_fill * 2
        self._fill_hull_vertices()

        ## this is only used for abspy construction
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

