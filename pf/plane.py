import open3d as o3d
import numpy as np
from dsrb import Berger
from pycompod import PyPlane
import os

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

ds = Berger()
model = ds.get_models(names="anchor",scan_configuration="1")[0]
data = np.load(model["eval"]["occ"])
points = data["points"]
occs = np.unpackbits(data["occupancies"])

class_weights = {0: 0.1, 1: 1}

svc = make_pipeline(StandardScaler(),
                    LinearSVC(random_state=0, class_weight=class_weights, C=1,penalty="l2",dual=False))
# ss = StandardScaler()
# points = ss.fit_transform(points)

# svc = LinearSVC(random_state=0, class_weight=class_weights)

svc = svc.named_steps['linearsvc']
svc.fit(points, occs)
plane_eq = np.hstack((svc.coef_,svc.intercept_[np.newaxis,:])).flatten()

plane = PyPlane(params=plane_eq)

side = plane.eval_points(points,return_sign=True)
count = (side==-1).sum()
print("Sides: {}/{}".format(100000-count,count))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
colors = np.zeros(shape=points.shape) + [0, 0, 1]
colors[occs.astype(bool)] = [1,0,0]
pcd.colors = o3d.utility.Vector3dVector(colors)

outpath = "/home/rsulzer/Downloads/occ_plane/"
os.makedirs(outpath,exist_ok=True)
outfile = os.path.join(outpath,"points.ply")
o3d.io.write_point_cloud(outfile,pcd,write_ascii=True)

plane_mesh = plane.get_trimesh_of_projected_points(np.asarray(pcd.get_axis_aligned_bounding_box().get_box_points()))
outfile = os.path.join(outpath,"plane.ply")
plane_mesh.export(outfile)





a=5


