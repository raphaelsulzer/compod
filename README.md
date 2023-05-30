## Introduction


Compact Polyhedral Complex for Surface Reconstruction and Convex Decomposition 

## Installation

### Install requirements

All dependencies except for [SageMath](https://www.sagemath.org/) can be easily installed with [PyPI](https://pypi.org/):

```bash
git clone https://github.com/chenzhaiyu/abspy && cd abspy
pip install -r requirements.txt
```

Optionally, install [pyglet](https://github.com/pyglet/pyglet) and [pyembree](https://github.com/adam-grant-hendry/pyembree) for better visualisation and ray-tracing, respectively:

```bash
pip install pyglet pyembree
```

### Install SageMath

For Linux and macOS users, the easiest is to install from [conda-forge](https://conda-forge.org/):

```bash
conda config --add channels conda-forge
conda install sage
```

Alternatively, you can use [mamba](https://github.com/mamba-org/mamba) for faster parsing and package installation:

```bash
conda install mamba
mamba install sage
```



### Install compod


## Quick start

Here is an example of loading a point cloud in `VertexGroup` (`.vg`), partitioning the ambient space into candidate convexes, creating the adjacency graph, and extracting the outer surface of the object.

