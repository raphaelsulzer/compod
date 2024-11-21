import os, sys
from glob import glob
from tqdm import tqdm

from pycompod import VertexGroup, PolyhedralComplex, make_logger


class BuildingReconstructor(PolyhedralComplex):






if __name__ == "__main__":
    # inherit from COMPOD, and during splitting assign the occupancy points to the corresponding child nodes

    verbosity = 20

    data_path = "/home/rsulzer/data/ign/compocity_test"
    in_path = os.path.join(data_path,"planes")

    buildings = glob(in_path+"/*.npz")

    buildings = [buildings[0]]

    logger = make_logger("Compocity",verbosity)

    for infile in tqdm(buildings,file=sys.stdout,disable=logger.level>30,position=0,leave=True):

        model = infile.split("/")[-1].split(".")[0]

        vg = VertexGroup(infile, verbosity=verbosity, debug_export=True)
        vg.classes = None

        cc = PolyhedralComplex(vg, device='gpu', verbosity=verbosity)
        cc.construct_partition()
        cc.add_bounding_box_planes()
        cc.label_partition(mode="normals")
        # ## needs compose extension
        # cc.label_partition(mode="mesh",mesh_file="data/{}/surface/dense_mesh.off".format(model))

        cc.simplify_partition_tree_based()
        cc.save_partition(os.path.join(data_path,"partition","partition.ply"), export_boundary=True)
        cc.save_partition_to_pickle(os.path.join(data_path,"partition","partition"))

        ## needs compose extension
        cc.save_simplified_surface(os.path.join(data_path,"surface","{}.obj".format(model)), triangulate=False,
                                   backend="wavefront")

        a=5
