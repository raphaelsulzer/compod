import os

from pycompod import PolyhedralComplex, VertexGroup






model = "bunny"


file = "../../../cpp/psdr/example/data/{}/convexes_detected/file.npz".format(model)
vg = VertexGroup(file)



# TOOD: remove the model parameter

cc = PolyhedralComplex(vg,device='gpu')

cc.construct_partition()
cc.add_bounding_box_planes()

os.makedirs("data/{}/partition".format(model),exist_ok=True)
cc.save_partition_to_pickle("data/{}/partition".format(model))

cc.save_partition("data/{}/partition/file.ply".format(model), rand_colors=False,
                  export_boundary=True, with_primitive_id=False)

cc.label_partition(mesh_file="data/{}/dense_mesh/file.off".format(model),graph_cut=False,type="mesh",outpath="data/{}/partition".format(model))


# cc.load_partition_from_pickle("data/{}/partition".format(model))
#
# occ_file = os.path.join("data/{}/partition".format(model),"occupancies.npz")
# cc.label_partition(mesh_file="data/{}/dense_mesh/file.off".format(model),graph_cut=False,type="load",out_path="data/{}/partition".format(model))

cc.simplify_partition_tree_based()
cc.simplify_partition_graph_based()

cc.save_in_cells(out_file="data/{}/in_cells/file.ply".format(model))

cc.save_surface(out_file="data/{}/polygon_mesh/file.off".format(model), backend="python", triangulate=False)

