# COMPOD: Compact Polyhedral Decomposition from Polygons

One GIF of rotating bunny. Start with point cloud, detect polygons, insert polygon one by one in complex, extract decomposition, 
make it explode, put it back together and show compact surface. 


This repository contains the official implementation of COMPOD: Compact Polyhedral Decomposition from Polygons.
Given a set of input polygons the resulting polyhedral complex can be used for polygon mesh reconstruction and convex decomposition. 


# Features

- Reading of vertex groups ([.vg](https://abspy.readthedocs.io/en/latest/vertexgroup.html), .npz) as input (e.g. from [here](https://github.com/raphaelsulzer/psdr/tree/main))
- Fast and memory efficient compact polyhedral complex construction (see evaluation)
- Storing of the complex as a binary space partitioning tree (BSP-tree)
- Interior / exterior labelling of the complex using point normals or a closed surface mesh
- Further simplification of the complex based on a careful analysis of the BSP-tree 
- Extraction of a compact convex decomposition (i.e. interior cells of the complex), or a compact polygon surface mesh (i.e. interface polygons between interior amd exterior cells of the complex). 

# Installation

Simply clone the repository and install in a new conda environment using pip:

```
git clone https://github.com/raphaelsulzer/compod.git
cd compod
conda create --name compod
conda activate compod
bash install.sh    # this step may take some time
pip install . 
```

You are now ready to use COMPOD.

## COMPOSE

COMPOSE is an extension for COMPOD that implements some routines for Surface Extraction in C++. Those are:
- a fast inside/outside labelling of the cells of the polyhedral complex based on sampling points in a reference mesh. 
- a simplification of the surface extracted from COMPOD based on a Constrained Delaunay Triangulation of the corner vertices of each planar region of the surface mesh.

To install COMPOSE you need to follow the steps below:

```
cd compose
conda install -y -c conda-forge spdlog cgal anaconda::mpfr
pip install . 
```




# Usage

```
from pypsdr import psdr

# initialise a planar shape detector and load input points                                              
ps = psdr(verbosity=1)                                               
ps.load_points(example/data/anchor/pointcloud.ply)

# detect planar shapes with default values
ps.detect(min_inliers=20,epsilon=0.02,normal_th=0.8,knn=10)

# refine planar shape configuration until convergence (i.e. no limit on number of iterations)
ps.refine(max_iter=-1)

# export planar shapes and vertex groups  
ps.save(example/data/anchor/convexes.ply,"convex")                  
ps.save(example/data/anchor/rectangles.ply,"rectangles")            
ps.save(example/data/anchor/alpha_shapes.ply,"alpha")               
ps.save(example/data/anchor/groups.vg)                              
ps.save(example/data/anchor/groups.npz)                             
```

# Examples

Please see the `example/` folder.

<p float="left">
  <img style="width:800px;" src="./media/city.gif">
</p>



# References

If you use this work please consider citing:

```bibtex
@article{1,
  title={Creating large-scale city models from 3D-point clouds: a robust approach with hybrid representation},
  author={Lafarge, Florent and Mallet, Cl{\'e}ment},
  journal={International journal of computer vision},
  volume={99},
  pages={69--85},
  year={2012},
  publisher={Springer}
}
```
