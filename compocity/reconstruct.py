import os, sys
from glob import glob
from tqdm import tqdm
from pypsdr import psdr
import numpy as np
import open3d as o3d
from pycompod import make_logger

class PlaneExtractor:

    def __init__(self,path,debug_path="",verbosity=20):

        self.path = path
        self.debug_path = debug_path
        self.verbosity = verbosity

        self.logger = make_logger(name="PlaneExtractor",level=self.verbosity)

    def extract_planes_from_lidar_points(self,infile,outfile):

        building = np.load(infile)

        classification = building["lidar_classification"]
        mask = classification == 6
        points = building["lidar_points"][mask.flatten()]
        normals = building["lidar_normals"][mask.flatten()]

        # initialise a planar shape detector
        ps = psdr(verbosity=1)

        # load input point cloud
        ps.load_points(points, normals)
        bb_diagonal = ps.get_bounding_box_diagonal()

        # detect planar shapes with fitting tolerance epsilon = 1% of the pointcloud's bounding box diagonal
        ps.detect(epsilon=0.01 * bb_diagonal, min_inliers=50, knn=10, normal_th=0.8)

        ps.save(outfile)


    def sample_wall_points(self,pt_min,pt_max,step_size=0.4):

        x_min = pt_min[0]
        y_min = pt_min[1]
        z_min = pt_min[2]
        x_max = pt_max[0]
        y_max = pt_max[1]
        z_max = pt_max[2]

        # Define the corner points of the 3D bounding box
        A = np.array([x_min, y_min, z_min])  # Min corner
        B = np.array([x_max, y_max, z_min])
        C = np.array([x_min, y_min, z_max])# Max corner

        # Compute the number of steps dynamically
        n_xy = int(np.ceil(np.linalg.norm(B - A) / step_size))

        n_z = int(np.ceil((z_max - z_min) / step_size))

        # Generate the sampling points along each axis
        pts_xy = np.linspace(A,B, n_xy)
        pts_z = np.linspace(0,z_max-z_min, n_z)

        points = np.repeat(pts_xy, pts_z.shape[0], axis=0)
        pts_z = np.tile(pts_z, pts_xy.shape[0])

        points[:, 2] += pts_z


        # make the normals
        v1 = B - A
        v2 = C - A

        normal = np.cross(v1,v2)
        normal = normal/ np.linalg.norm(normal)
        normals = np.tile(normal,(points.shape[0],1))

        return points,normals


    def extract_planes_from_footprints(self,infile_building,infile_planes):

        building = np.load(infile_building)
        bbox = building["bbox"]
        z_min = bbox[0,2]
        z_max= bbox[0,5]

        sampled_points = []
        sampled_normals = []
        sampled_planes = []

        polys = building["footprint_poly_size"].flatten()
        poly_points = building["footprint_points"]

        prev = 0
        for poly in polys:

            this_points = poly_points[prev:poly]

            for i in range(len(this_points)):
                start = this_points[i]
                end = this_points[(i+1)%len(this_points)]

                start = np.hstack((start,z_min))
                end = np.hstack((end,z_max))

                points, normals = self.sample_wall_points(start,end)

                sampled_points.append(points)
                sampled_normals.append(normals)

            prev = poly


        sampled_points = np.concatenate(sampled_points)
        sampled_normals = np.concatenate(sampled_normals)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(sampled_points)
        pcd.normals = o3d.utility.Vector3dVector(sampled_normals)
        outfile = os.path.join(self.debug_path,"points.ply")
        os.makedirs(os.path.dirname(outfile),exist_ok=True)
        o3d.io.write_point_cloud(outfile, pcd)









if __name__ == "__main__":


    # infile = "/home/rsulzer/data/ign/compocity_test/footprints_all/BATIMENT0000000327904050.npz"

    data_path = "/home/rsulzer/data/ign/compocity_test"
    in_path = os.path.join(data_path,"footprints_all")
    out_path = os.path.join(data_path,"planes")

    pe = PlaneExtractor(data_path,debug_path=os.path.join(data_path,"DEBUG"))


    buildings = glob(in_path+"/*.npz")

    buildings = [buildings[0]]

    for infile in tqdm(buildings,file=sys.stdout,disable=pe.logger.level>30,position=0,leave=True):

        pe.logger.info("Process building {}".format(infile))

        outfile = os.path.join(out_path,os.path.split(infile)[1])
        # extract_planes_from_lidar_points(infile,outfile)

        pe.extract_planes_from_footprints(infile,outfile)









