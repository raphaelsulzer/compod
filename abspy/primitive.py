import os, sys, struct
from pathlib import Path
import numpy as np
import torch
from sage.all import polytopes, QQ, RR, Polyhedron
# from torchrec import sparse
PYTHONPATH="/home/rsulzer/python"
sys.path.append(os.path.join(PYTHONPATH,"pyplane"))
from pyplane import PyPlane, SagePlane, ProjectedConvexHull
from export import PlaneExporter

from .logger import attach_to_log

logger = attach_to_log()


class VertexGroup:
    """
    Class for manipulating planar primitives.
    """

    def __init__(self, filepath, merge_duplicates=False, prioritise_planes = None,
                 points_type="inliers", sample_count_per_area=2, fixed_sample_count=10, export=False,
                 backend='gpu'):
        """
        Init VertexGroup.
        Class for manipulating planar primitives.

        Parameters
        ----------
        filepath: str or Path
            Filepath to vertex group file (.vg) or binary vertex group file (.bvg)
        """

        if isinstance(filepath, str):
            self.filepath = Path(filepath)
        else:
            self.filepath = filepath
        self.processed = False
        self.points = None
        self.planes = None
        self.halfspaces = []
        self.bounds = None
        self.points_grouped = None
        self.points_ungrouped = None
        self.merge_duplicates = merge_duplicates
        self.prioritise_planes = prioritise_planes
        self.sample_count_per_area = sample_count_per_area
        self.fixed_sample_count = fixed_sample_count

        self.points_type = points_type

        self.export = export
        self.backend = backend

        ending = os.path.splitext(filepath)[1]
        if ending == ".npz":
            self._process_npz()
        elif ending == ".vg":
            self._process()
            del self.lines # for closing the .vg file
        else:
            print("{} is not a valid file type for planes".format(ending))
            sys.exit(1)



    def _load_vg_file(self):
        """
        Load (ascii / binary) vertex group file.
        """
        if self.filepath.suffix == '.vg':
            with open(self.filepath, 'r') as fin:
                # self.lines=np.array(fin.readlines())
                # self.lines=np.array(fin.readlines())
                self.lines = np.array(list(fin))


        elif self.filepath.suffix == '.bvg':
            # define size constants
            _SIZE_OF_INT = 4
            _SIZE_OF_FLOAT = 4
            _SIZE_OF_PARAM = 4
            _SIZE_OF_COLOR = 3

            vgroup_ascii = ''
            with open(self.filepath, 'rb') as fin:
                # points
                num_points = struct.unpack('i', fin.read(_SIZE_OF_INT))[0]
                points = struct.unpack('f' * num_points * 3, fin.read(_SIZE_OF_FLOAT * num_points * 3))
                vgroup_ascii += f'num_points: {num_points}\n'
                vgroup_ascii += ' '.join(map(str, points)) + '\n'

                # colors
                num_colors = struct.unpack("i", fin.read(_SIZE_OF_INT))[0]
                vgroup_ascii += f'num_colors: {num_colors}\n'

                # normals
                num_normals = struct.unpack("i", fin.read(_SIZE_OF_INT))[0]
                normals = struct.unpack('f' * num_normals * 3, fin.read(_SIZE_OF_FLOAT * num_normals * 3))
                vgroup_ascii += f'num_normals: {num_normals}\n'
                vgroup_ascii += ' '.join(map(str, normals)) + '\n'

                # groups
                num_groups = struct.unpack("i", fin.read(_SIZE_OF_INT))[0]
                vgroup_ascii += f'num_groups: {num_groups}\n'

                group_counter = 0
                while group_counter < num_groups:
                    group_type = struct.unpack("i", fin.read(_SIZE_OF_INT))[0]
                    num_group_parameters = struct.unpack("i", fin.read(_SIZE_OF_INT))[0]
                    group_parameters = struct.unpack("f" * _SIZE_OF_PARAM, fin.read(_SIZE_OF_INT * _SIZE_OF_PARAM))
                    group_label_size = struct.unpack("i", fin.read(_SIZE_OF_INT))[0]
                    # be reminded that vg <-> bvg in Mapple does not maintain group order
                    group_label = struct.unpack("c" * group_label_size, fin.read(group_label_size))
                    group_color = struct.unpack("f" * _SIZE_OF_COLOR, fin.read(_SIZE_OF_FLOAT * _SIZE_OF_COLOR))
                    group_num_point = struct.unpack("i", fin.read(_SIZE_OF_INT))[0]
                    group_points = struct.unpack("i" * group_num_point, fin.read(_SIZE_OF_INT * group_num_point))
                    num_children = struct.unpack("i", fin.read(_SIZE_OF_INT))[0]

                    vgroup_ascii += f'group_type: {group_type}\n'
                    vgroup_ascii += f'num_group_parameters: {num_group_parameters}\n'
                    vgroup_ascii += 'group_parameters: ' + ' '.join(map(str, group_parameters)) + '\n'
                    vgroup_ascii += 'group_label: ' + ''.join(map(str, group_label)) + '\n'
                    vgroup_ascii += 'group_color: ' + ' '.join(map(str, group_color)) + '\n'
                    vgroup_ascii += f'group_num_point: {group_num_point}\n'
                    vgroup_ascii += ' '.join(map(str, group_points)) + '\n'
                    vgroup_ascii += f'num_children: {num_children}\n'

                    group_counter += 1

                # convert vgroup_ascii to list
                return vgroup_ascii.split('\n')

        else:
            raise ValueError(f'unable to load {self.filepath}, expected *.vg or .bvg.')

    def _process(self):
        """
        Start processing vertex group.
        """
        logger.debug('processing {}'.format(self.filepath))
        self._load_vg_file()
        self.points = self._get_points()
        self.planes, self.bounds, self.points_grouped, self.points_ungrouped = self._get_primitives()
        self.processed = True


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
        logger.info('Prioritise planar primitive with mode {}'.format(mode))

        indices_sorted_planes = np.arange(len(self.planes))

        if mode == "random":
            np.random.shuffle(indices_sorted_planes)
            return indices_sorted_planes
        elif mode == "vertical":
            indices_vertical_planes = self._vertical_planes(slope_threshold=0.9)
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


    def _fill_hull_vertices(self):

        hull_vertices = []

        for i,v in enumerate(self.hull_vertices):

            hp = np.array(self.convex_hulls[i].hull_points)
            fill_vertices = hp[np.random.choice(hp.shape[0],self.n_fill-hp.shape[0])]
            hp = np.concatenate((hp,fill_vertices))
            hull_vertices.append(hp)


        if self.backend == 'gpu':
            # self.hull_vertices = torch.HalfTensor(np.array(hull_vertices)).to('cuda')
            self.hull_vertices = torch.Tensor(np.array(hull_vertices)).to('cuda')
        elif self.backend == 'cpu':
            self.hull_vertices = np.array(hull_vertices)
        else:
            raise NotImplementedError




    def _sample_polygons(self,n_points=None):

        ## project inliers to plane and get the convex hull
        all_sampled_points = []
        for i,poly in enumerate(self.polygons):
            np.random.seed(42)
            if n_points is None:
                n = 3+int(self.sample_count_per_area*poly.area)
            sampled_points = poly.sample(n)
            sampled_points = np.concatenate((sampled_points,poly.vertices),axis=0,dtype=np.float32)
            all_sampled_points.append(sampled_points)
            self.convex_hulls[i].all_points = sampled_points

        self.points_grouped = np.array(all_sampled_points,dtype=object)



    def _process_npz(self):
        """
        Start processing vertex group.
        """


        fn = self.filepath.with_suffix(".npz")
        data = np.load(fn)

        # read the data and make the point groups
        self.planes = data["group_parameters"].astype(np.float32)
        points = data["points"].astype(np.float32)
        npoints = data["group_num_points"].flatten()
        verts = data["group_points"].flatten()
        self.plane_colors = data["group_colors"]
        self.polygons = []
        self.polygon_areas = []
        self.points_grouped = []
        n_hull_points = []
        self.convex_hulls = []
        self.hull_vertices = []
        last = 0
        for i,npp in enumerate(npoints):
            ## make the point groups
            vert_group = verts[last:(npp+last)]
            pts = points[vert_group]
            self.points_grouped.append(pts)

            # TODO: i am computing the convex hull twice below; not necessary

            ## make the polys
            ## make a trimesh of each input polygon
            pl = PyPlane(self.planes[i])
            poly = pl.get_trimesh_of_projected_points(pts,type="convex_hull")
            self.polygons.append(poly)
            self.polygon_areas.append(poly.area)

            pch = ProjectedConvexHull(self.planes[i],pts)
            self.convex_hulls.append(pch)
            self.hull_vertices.append(pch.hull_points)
            n_hull_points.append(len(pch.hull_points))

            last += npp



        self.polygons = np.array(self.polygons)
        self.polygon_areas = np.array(self.polygon_areas)
        self.convex_hulls = np.array(self.convex_hulls)
        # fill the hull array to make it a matrix instead of jagged array for an efficient _get_best_plane function with matrix multiplications
        n_hull_points = np.array(n_hull_points)
        self.n_fill = n_hull_points.max()*2
        self._fill_hull_vertices()

        ### scale sample_count_per_area by total area of input polygons. like this n_sample_points should roughly be constant for each mesh + (convex hull points)
        self.sample_count_per_area = self.sample_count_per_area/self.polygon_areas.sum()

        if self.points_type == "samples":
            if self.fixed_sample_count:
                n_points = self.fixed_sample_count
            else:
                n_points = None
            self._sample_polygons(n_points=n_points) # redefines self.points_grouped
        elif self.points_type == "inliers":
            raise NotImplementedError
        else:
            print("{} is not a valid point_type. Only 'inliers' or 'samples' are allowed.".format(self.points_type))
            NotImplementedError



        if self.prioritise_planes:
            order = self._prioritise_planes(self.prioritise_planes)
            self.planes = self.planes[order]
            self.points_grouped = list(np.array(self.points_grouped,dtype=object)[order])
            self.polygons = self.polygons[order]
            self.polygon_areas = self.polygon_areas[order]
            self.hull_vertices = self.hull_vertices[order.copy(),:,:]
            self.convex_hulls = self.convex_hulls[order]
            self.plane_colors = self.plane_colors[order]
        else:
            logger.info("No plane prioritisation applied")




        ## export planes and samples
        pe = PlaneExporter()
        # pt_file = os.path.join(os.path.dirname(str(self.filepath)),"polygon_samples.ply")
        # plane_file =  os.path.join(os.path.dirname(str(self.filepath)),"merged_planes.ply")
        pt_file = os.path.splitext(str(self.filepath))[0]+"_samples.ply"
        plane_file =  self.filepath.with_suffix('.ply')
        pe.export_points_and_planes([pt_file,plane_file],self.points_grouped,self.planes,colors=self.plane_colors)



        n_planes = self.planes.shape[0]
        # merge input polygons that come from the same plane but are disconnected
        # this is desirable for the adaptive tree construction, because it otherwise may insert the same plane into the same cell twice
        if self.merge_duplicates:
            raise NotImplementedError # need to add the polygons tensor to this first
            # pts = defaultdict(int)
            # primitive_ids = defaultdict(list)
            # polygons = defaultdict(list)
            # cols = defaultdict(list)
            # un, inv = np.unique(self.planes, return_inverse=True, axis=0)
            # for i in range(len(self.planes)):
            #     if isinstance(pts[inv[i]],int): ## hacky way to check if this item already has a value or is empty, ie has the default int assigned
            #         pts[inv[i]] = self.points_grouped[i]
            #         polygons[inv[i]] = self.polygons[i]
            #     else:
            #         pts[inv[i]] = np.concatenate((pts[inv[i]],self.points_grouped[i]))
            #         polygons[inv[i]]+=self.polygons[i]
            #         # pp = polygons[inv[i]].vertices_list() + self.polygons[i].vertices_list()
            #         # polygons[inv[i]] = Polyhedron(vertices=pp)
            #
            #     cols[inv[i]] = colors[i]
            #     primitive_ids[inv[i]]+=[i]
            #
            # self.plane_colors = list(cols.values())
            # self.planes = un[list(pts.keys())]
            # self.points_grouped = list(pts.values())
            # self.polygons = polygons
            # # put in the id of the merged primitive, ie also the plane, and get out the 1 to n input primitives that were merged for it
            # self.merged_primitives_to_input_primitives = list(primitive_ids.values())
            #
            # logger.info("Merged primitives from the same plane, reducing primitive count from {} to {}".format(n_planes,self.planes.shape[0]))
        else:
            self.merged_primitives_to_input_primitives = []
            for i in range(len(self.planes)):
                self.merged_primitives_to_input_primitives.append([i])


        self.bounds = []
        for pg in self.points_grouped:
            self.bounds.append(self._points_bound(pg))
        self.bounds = np.array(self.bounds)


        # make the bounds and halfspace used in the cell complex construction
        self.halfspaces = []
        self.plane_dict = dict()
        for i,p in enumerate(self.planes):
            self.plane_dict[str(p)] = i
            self.halfspaces.append([Polyhedron(ieqs=[inequality]) for inequality in self._inequalities(p)])
        self.halfspaces = np.array(self.halfspaces)

        self.points_ungrouped = np.zeros(points.shape[0])
        self.points_ungrouped[verts] = 1
        self.points_ungrouped = np.invert(self.points_ungrouped.astype(bool))
        self.points_ungrouped = points[self.points_ungrouped.astype(int)]

        self.processed = True



    def _get_points(self):

        npoints = int(self.lines[0].split(':')[1])
        return np.genfromtxt(self.lines[1:npoints+1])

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

    def _get_primitives(self):
        """
        Get primitives from vertex group.

        Returns
        ----------
        params: (n, 4) float
            Plane parameters
        bounds: (n, 2, 3) float
            Bounding box of the primitives
        groups: (n, m, 3) float
            Groups of points
        ungrouped_points: (u, 3) float
            Points that belong to no group
        """
        # is_primitive = [line.startswith('group_num_point') for line in self.vgroup_ascii]
        is_primitive = [line.startswith('group_type') for line in self.lines]

        primitives = self.lines[np.roll(is_primitive,6)]
        # primitives = [self.lines[line] for line in np.where(is_primitive)[0] + 6]
        params = self.lines[np.roll(is_primitive,2)]


        # lines of groups in the file
        params_list = []
        bounds = []
        groups = []
        grouped_indices = set()  # indices of points being grouped
        for i, p in enumerate(primitives):
            point_indices = np.fromstring(p, sep=' ').astype(np.int64)
            grouped_indices.update(point_indices)
            points = self.points[point_indices]
            #### this is for fitting planes, which was in original code, but now I just use the plane equations
            # param = self.fit_plane(points, mode='PCA')
            # if param is None:
            #     continue
            # params.append(param)
            params_list.append(np.fromstring(params[i].split(':')[1],sep=' '))
            bounds.append(self._points_bound(points))
            groups.append(points)
        ungrouped_indices = set(range(len(self.points))).difference(grouped_indices)
        ungrouped_points = self.points[list(ungrouped_indices)]  # points that belong to no groups
        return np.array(params_list), np.array(bounds), np.array(groups, dtype=object), np.array(ungrouped_points)

    @staticmethod
    def _points_bound(points):
        """
        Get bounds (AABB) of the points.

        Parameters
        ----------
        points: (n, 3) float
            Points
        Returns
        ----------
        as_float: (2, 3) float
            Bounds (AABB) of the points
        """
        return np.array([np.amin(points, axis=0), np.amax(points, axis=0)])
