import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn.linear_model import Ridge
from scipy.optimize import minimize
from dsrb import Berger
from pycompod import PyPlane
import os
import open3d as o3d
import matplotlib
matplotlib.use('TkAgg')



# Define the optimization function to minimize the distance to occupied points
def objective(params):
    w = params[:3]
    b = params[3]
    distances = np.abs(np.dot(X_scaled, w) + b) / np.linalg.norm(w)
    return np.sum(distances[y == 0])  # Only minimize distance for empty points
# Define the constraints for occupied points
def constraint(params, X_scaled, y):
    w = params[:3]
    b = params[3]
    return np.dot(X_scaled[y == 1], w) + b - 1  # Ensure occupied points are correctly classified with a margin


def find_plane(X_scaled,y):
    # Initial guess for the plane parameters
    initial_params = np.random.rand(4)

    constraints = {'type': 'ineq', 'fun': constraint, 'args': (X_scaled, y)}

    # Perform the optimization
    result = minimize(objective, initial_params, method='SLSQP', constraints=constraints)

    # Get the optimized coefficients (weight vector) and intercept (bias term)
    w = result.x[:3]
    b = result.x[3]

    plane = PyPlane(params=result.x)
    side = plane.eval_points(X_scaled,return_sign=True)
    count = (side==-1).sum()
    print("Sides: {}/{}".format(len(X_scaled)-count,count))
    # TODO: add here the code that checks which side a point is on, remove all the empty points, and rerun the algo, to see the next plane


    print(f'Weight vector: {w}')
    print(f'Bias term: {b}')

    # Print the hyperplane equation
    print(f'Hyperplane equation: {w[0]} * x1 + {w[1]} * x2 + {w[2]} * x3 + {b} = 0')

    # sub = np.random.choice(np.arange(X_scaled.shape[0]), 5000, replace=False)
    # X_scaled = X_scaled[sub]
    # y=y[sub]
    #
    # # Visualize the data and the decision boundary
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], c=y, cmap='bwr', alpha=0.7)
    #
    # # Create a mesh grid for plotting the decision boundary
    # xx, yy = np.meshgrid(np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 50),
    #                      np.linspace(X_scaled[:, 1].min(), X_scaled[:, 1].max(), 50))
    #
    # # Calculate the corresponding z values for the decision boundary
    # zz = (-w[0] * xx - w[1] * yy - b) / w[2]
    #
    # # Plot the decision boundary
    # ax.plot_surface(xx, yy, zz, color='yellow', alpha=0.5)
    #
    # plt.show()

    return plane,side

def export_to_ply(points,occs,plane,count):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.zeros(shape=points.shape) + [0, 0, 1]
    colors[occs.astype(bool)] = [1, 0, 0]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    outpath = "/home/rsulzer/Downloads/occ_plane/"
    os.makedirs(outpath, exist_ok=True)
    outfile = os.path.join(outpath, "{}_points.ply".format(count))
    o3d.io.write_point_cloud(outfile, pcd, write_ascii=True)

    plane_mesh = plane.get_trimesh_of_projected_points(np.asarray(pcd.get_axis_aligned_bounding_box().get_box_points()))
    outfile = os.path.join(outpath, "{}_plane.ply".format(count))
    plane_mesh.export(outfile)


if __name__ == '__main__':
    # Create a sample 3D dataset
    # X, y = make_classification(n_samples=100000, n_features=3, n_informative=3, n_redundant=0, n_clusters_per_class=1, class_sep=0.5, random_state=42)

    ds = Berger()
    model = ds.get_models(names="anchor", scan_configuration="1")[0]
    data = np.load(model["eval"]["occ"])
    X = data["points"]
    y = np.unpackbits(data["occupancies"])

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    for i in range(50):
        plane, side = find_plane(X_scaled,y)
        export_to_ply(points=X_scaled,occs=y,plane=plane,count=i)
        remaining = side==1
        X_scaled = X_scaled[remaining]
        y = y[remaining]

        a=5

    # plane,side = find_plane(X_scaled,y)
    # export_to_ply(points=X_scaled,occs=y,plane=plane)

    a=5