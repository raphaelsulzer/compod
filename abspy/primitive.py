import os, sys, struct
from pathlib import Path
import numpy as np
from sage.all import polytopes, QQ, RR, Polyhedron
from collections import defaultdict

PYTHONPATH="/home/rsulzer/python"
sys.path.append(os.path.join(PYTHONPATH,"pyplane"))
from pyplane import PyPlane

from .logger import attach_to_log

logger = attach_to_log()


class VertexGroup:
    """
    Class for manipulating planar primitives.
    """

    def __init__(self, filepath, merge_duplicates=False, prioritise_planes = None, points_type="inliers", sample_count_per_area=2):
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

        self.points_type = points_type

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

    def _sample_polygons(self,planes,points,n_points_per_area=2):

        ## project inliers to plane and get the convex hull
        all_sampled_points = []
        for i,plane in enumerate(planes):

            pl = PyPlane(plane)
            mesh=pl.get_trimesh_of_projected_points(points[i])
            sampled_points = mesh.sample(2+int(n_points_per_area*mesh.area))
            all_sampled_points.append(sampled_points)

        return np.array(all_sampled_points,dtype=object)


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
        elif mode == "n_points":
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
            areas = []
            for i,plane in enumerate(self.planes):
                mesh = PyPlane(plane).get_trimesh_of_projected_points(self.points_grouped[i])
                areas.append(mesh.area)
            return np.argsort(areas)[::-1]
        else:
            NotImplementedError



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



    def _process_npz(self):
        """
        Start processing vertex group.
        """

        fn = self.filepath.with_suffix(".npz")
        data = np.load(fn)

        # read the data and make the point groups
        self.planes = data["group_parameters"]
        points = data["points"]
        npoints = data["group_num_points"].flatten()
        verts = data["group_points"].flatten()
        colors = data["group_colors"]
        self.points_grouped = []
        last = 0
        for npp in npoints:
            vert_group = verts[last:(npp+last)]
            self.points_grouped.append(points[vert_group])
            last += npp

        if self.points_type == "samples":
            self.points_grouped = self._sample_polygons(self.planes,self.points_grouped,self.sample_count_per_area)
            self.merge_duplicates = True
        elif self.points_type == "inliers":
            pass
        else:
            print("{} is not a valid point_type. Only 'inliers' or 'samples' are allowed.".format(self.points_type))
            NotImplementedError




        n_planes = self.planes.shape[0]
        # merge input polygons that come from the same plane but are disconnected
        # this is desirable for the adaptive tree construction, because it otherwise may insert the same plane into the same cell twice
        if self.merge_duplicates:
            self.bounds = []
            # d = defaultdict(list)
            pts = defaultdict(int)
            ids = defaultdict(list)
            cols = defaultdict(list)
            un, inv = np.unique(self.planes, return_inverse=True, axis=0)
            for i in range(len(self.planes)):
                if isinstance(pts[inv[i]],int): ## hacky way to check if this item already has a value or is empty, ie has the default int assigned
                    pts[inv[i]] = self.points_grouped[i]
                else:
                    pts[inv[i]] = np.concatenate((pts[inv[i]],self.points_grouped[i]))

                cols[inv[i]] = colors[i]
                ids[inv[i]]+=[i]

            self.plane_colors = list(cols.values())
            self.planes = un[list(pts.keys())]
            self.points_grouped = list(pts.values())
            # put in the id of the merged primitive, ie also the plane, and get out the 1 to n input primitives that were merged for it
            self.merged_primitives_to_input_primitives = list(ids.values())

            logger.info("Merged primitives from the same plane, reducing primitive count from {} to {}".format(n_planes,self.planes.shape[0]))
        else:
            self.plane_colors = colors
            self.merged_primitives_to_input_primitives = []
            for i in range(len(self.planes)):
                self.merged_primitives_to_input_primitives.append([i])

        self.bounds = []
        for pg in self.points_grouped:
            self.bounds.append(self._points_bound(pg))
        self.bounds = np.array(self.bounds)

        if self.prioritise_planes:
            order = self._prioritise_planes(self.prioritise_planes)
            self.planes = self.planes[order]
            self.points_grouped = list(np.array(self.points_grouped,dtype=object)[order])
            self.merged_primitives_to_input_primitives = list(np.array(self.merged_primitives_to_input_primitives,dtype=object)[order])
            self.plane_colors = np.array(self.plane_colors)[order]
            self.bounds = self.bounds[order]
        else:
            logger.info("No plane prioritisation applied")

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
