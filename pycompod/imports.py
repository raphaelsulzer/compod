### form import the C++ executables
import pathlib, sys, os
pp = pathlib.Path(__file__).parents[3]
CPPPATH=str(os.path.join(pp,"cpp"))
MODE = "debug"
MODE = "release"
sys.path.append(os.path.join(CPPPATH,"compact_mesh_reconstruction/build/{}/Benchmark/PyLabeler".format(MODE)))
sys.path.append(os.path.join(CPPPATH,"compact_mesh_reconstruction/build/{}/Benchmark/Soup2Mesh".format(MODE)))
sys.path.append(os.path.join(CPPPATH,"compact_mesh_reconstruction/build/{}/Benchmark/SimplifySurface".format(MODE)))

