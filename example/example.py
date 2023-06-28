import os
from pycompod import PolyhedralComplex, VertexGroup



model = "anchor"


file = "../../../cpp/psdr/example/data/{}/convexes_detected/file.npz".format(model)
vg = VertexGroup(file,prioritise="area")


cc = PolyhedralComplex(vg,device='gpu',logging_level=20)

cc.construct_partition()
cc.add_bounding_box_planes()
cc.label_partition(mesh_file="data/{}/dense_mesh/file.off".format(model),graph_cut=False,type="mesh")

os.makedirs("data/{}/partition".format(model),exist_ok=True)

cc.save_partition("data/{}/partition/file.ply".format(model), rand_colors=False,
                  export_boundary=True, with_primitive_id=False)


cc.simplify_partition_tree_based()
# cc.simplify_partition_graph_based()

cc.save_partition_to_pickle("data/{}/partition".format(model))

cc.load_partition_from_pickle("data/{}/partition".format(model))


cc.save_in_cells(out_file="data/{}/in_cells/file.ply".format(model))
cc.save_surface(out_file="data/{}/polygon_mesh/file.off".format(model), backend="python", triangulate=False)
cc.save_simplified_surface(out_file="data/{}/triangle_mesh/file.off".format(model), backend="cgal", triangulate=False)

