from pycompod import VertexGroup, PolyhedralComplex

model = "anchor" # or "bunny"

file = "data/{}/convexes_refined/file.npz".format(model)
vg = VertexGroup(file,verbosity=20)
cc = PolyhedralComplex(vg,device='gpu',verbosity=20)

cc.construct_partition()
cc.add_bounding_box_planes()
cc.label_partition(mode="normals")
# ## needs compose extension
# cc.label_partition(mode="mesh",mesh_file="data/{}/surface/dense_mesh.off".format(model))

cc.simplify_partition_tree_based()
cc.save_partition("data/{}/partition/tree_simplified_partition.ply".format(model), export_boundary=True)
cc.save_partition_to_pickle("data/{}/partition".format(model))

cc.save_surface(out_file="data/{}/surface/complex_mesh.obj".format(model), triangulate=False)
## needs compose extension
cc.save_simplified_surface(out_file="data/{}/surface/polygon_mesh.obj".format(model), triangulate=False)
cc.save_simplified_surface(out_file="data/{}/surface/triangle_mesh.obj".format(model), triangulate=True)