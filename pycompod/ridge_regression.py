from sklearn.linear_model import Ridge
from sklearn.datasets import make_classification
import numpy as np
from sklearn.preprocessing import StandardScaler


import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use( 'tkagg' )
# Create a sample 3D dataset

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import open3d as o3d
import numpy as np
from dsrb import Berger
from pycompod import PyPlane
import os

ds = Berger()
model = ds.get_models(names="anchor",scan_configuration="1")[0]
data = np.load(model["eval"]["occ"])
points = data["points"]
occs = np.unpackbits(data["occupancies"])

scaler = StandardScaler()
points = scaler.fit_transform(points)

# Fit Ridge regression model


# Convert class labels from {0, 1} to {-1, 1} for ease of interpretation
occs = np.where(occs == 0, -1, 1)

# Fit Ridge regression model
ridge = Ridge(alpha=1.0)  # alpha is the regularization strength
ridge.fit(points, occs)

# Get the coefficients (weight vector) and intercept (bias term)
w = ridge.coef_.tolist()
b = ridge.intercept_

w.append(b)

plane_eq=np.array(w)

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


# print(f'Weight vector: {w}')
# print(f'Bias term: {b}')
#
# # Print the hyperplane equation
# print(f'Hyperplane equation: {w[0]} * x1 + {w[1]} * x2 + {w[2]} * x3 + {b} = 0')
#
# # Visualize the data and the decision boundary
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='bwr', alpha=0.7)
#
# # Create a mesh grid for plotting the decision boundary
# xx, yy = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 50),
#                      np.linspace(X[:, 1].min(), X[:, 1].max(), 50))
#
# # Calculate the corresponding z values for the decision boundary
# zz = (-w[0] * xx - w[1] * yy - b) / w[2]
#
# # Plot the decision boundary
# ax.plot_surface(xx, yy, zz, color='yellow', alpha=0.5)
#
# plt.show(block=True)