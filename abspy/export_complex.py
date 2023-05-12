import os
import numpy as np
import open3d as o3d
from pathlib import Path

class CellComplexExporter:
    """
    Class of cell complex from planar primitive arrangement.
    """
    def __init__(self, cellComplex):
        """
        Init CellComplex.
        Class of cell complex from planar primitive arrangement.

        Parameters
        ----------
        planes: (n, 4) float
            Plana parameters
        bounds: (n, 2, 3) float
            Corresponding bounding box bounds of the planar primitives
        points: (n, ) object of float
            Points grouped into primitives, points[any]: (m, 3)
        initial_bound: None or (2, 3) float
            Initial bound to partition
        build_graph: bool
            Build the cell adjacency graph if set True.
        additional_planes: None or (n, 4) float
            Additional planes to append to the complex,
            can be missing planes due to occlusion or incapacity of RANSAC
        quiet: bool
            Disable logging and progress bar if set True
        """
        self.cellComplex = cellComplex


    def write_graph(self, m, graph, cells, subfolder="", color = None):

        c = color if color is not None else np.random.random(size=3)
        c = (c*255).astype(int)

        path = os.path.join(os.path.dirname(m['planes']),subfolder)
        os.makedirs(path,exist_ok=True)
        filename = os.path.join(path,'graph.obj')

        edge_strings = []
        f = open(filename,'w')
        all_nodes = np.array(graph.nodes())
        for i,node in enumerate(graph.nodes(data=False)):
            centroid = np.array(cells.get(node).center())
            f.write("v {:.3f} {:.3f} {:.3f} {} {} {}\n".format(centroid[0],centroid[1],centroid[2],c[0],c[1],c[2]))
            edges = list(graph.edges(node[0]))
            for c1,c2 in edges:
                if not graph.edges[c1,c2]["convex_intersection"]:
                    nc1 = np.where(all_nodes == c1)[0][0]
                    nc2 = np.where(all_nodes == c2)[0][0]
                    edge_strings.append("l {} {}\n".format(nc1 + 1, nc2 + 1))
                # nc1 = np.where(all_nodes==c1)[0][0]
                # nc2 = np.where(all_nodes==c2)[0][0]
                # edge_strings.append("l {} {} {} {} {}\n".format(nc1+1,nc2+1,col[0],col[1],col[2]))


        for edge in edge_strings:
            f.write(edge)

        f.close()




    def write_cell(self, m, polyhedron, points=None, filename=None, subfolder="partitions",count=0, color=None, inside_vert_count=0, to_ply=False):

        c = color if color is not None else np.random.random(size=3)
        c = (c*255).astype(int)

        path = os.path.join(os.path.dirname(m['planes']),subfolder)
        os.makedirs(path,exist_ok=True)

        if filename is None:
            filename = os.path.join(path,str(count)+'.obj')
            f = open(filename,'w')
        else:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            f = open(filename, 'a')
            f.write('o {}\n'.format(count))

        ss = polyhedron.render_solid().obj_repr(polyhedron.render_solid().default_render_params())

        verts = ss[2]
        for v in verts:
            f.write(v + " {} {} {}\n".format(c[0],c[1],c[2]))
        faces = ss[3]
        for fa in faces:
            f.write(fa[0] + " ")
            for ffa in fa[2:].split(' '):
                f.write(str(int(ffa)+inside_vert_count)+" ")
            f.write("\n")

        if points is not None:
            for p in points:
                f.write("v {:.3f} {:.3f} {:.3f} {} {} {}\n".format(p[0],p[1],p[2],c[0],c[1],c[2]))

        f.close()





    def write_facet(self,m,facet,subfolder="facets",count=0, color=None):

        c = color if color is not None else np.random.random(size=3)
        c = (c*255).astype(int)

        path = os.path.join(os.path.dirname(m['planes']),subfolder)
        os.makedirs(path,exist_ok=True)
        filename = os.path.join(path,str(count)+'.obj')

        ss = facet.render_solid().obj_repr(facet.render_solid().default_render_params())

        f = open(filename,'w')
        verts = ss[2]
        for v in verts:
            f.write(v + " {} {} {}\n".format(c[0],c[1],c[2]))
        faces = ss[3]
        for fa in faces:
            f.write(fa+"\n")

        f.close()

    def write_graph_edge(self,m,graph,e0,e1):

        assert (len(graph[e0][e1]["vertices"]) > 2)

        pts = []
        for v in graph[e0][e1]["vertices"]:
            pts.append(tuple(v))
        pts = list(set(pts))
        intersection_points = np.array(pts, dtype=object)

        correct_order = self.cellComplex._sort_vertex_indices_by_angle(intersection_points.astype(float),
                                                     graph[e0][e1]["supporting_plane"])
        assert (len(intersection_points) == len(correct_order))
        intersection_points = intersection_points[correct_order]

        if (len(intersection_points) < 3):
            print("WARNING: graph edge with less than three polygon vertices")
            return

        ## orient triangle

        ## TODO: problem here is that orientation doesn't work when points are on the same line, because then e1 and e2 are coplanar
        outside = graph.nodes[e0]["convex"].centroid()
        ei1 = (intersection_points[1] - intersection_points[0]).astype(float)
        ei1 = ei1 / np.linalg.norm(ei1)
        ei2 = (intersection_points[-1] - intersection_points[0]).astype(float)
        ei2 = ei2 / np.linalg.norm(ei2)
        # e2 = e1
        # s=1
        # while np.isclose(np.arccos(np.dot(e1,e2)),0,rtol=1e-02):
        #     s+=1
        #     e2 = (intersection_points[s] - intersection_points[0]).astype(float)
        #     e2 = e2/np.linalg.norm(e2)
        ei3 = (outside - intersection_points[0]).astype(float)
        ei3 = ei3 / np.linalg.norm(ei3)
        if self.cellComplex._orient_triangle(ei1, ei2, ei3):
            intersection_points = np.flip(intersection_points, axis=0)

        id = graph[e0][e1]["id"]
        filename = os.path.join(os.path.dirname(m["planes"]),"graph_facets",str(id)+".off")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.write_off(filename, points=intersection_points.astype(float),facets=[np.arange(len(intersection_points))],color=graph[e0][e1]["color"])

    def write_facet_with_outside_centroid(self, m, points, outside, count=0):

        filename = os.path.join(os.path.dirname(m["planes"]),"facets_with_outside_centroid",str(count)+".obj")
        os.makedirs(os.path.dirname(filename),exist_ok=True)

        f = open(filename, 'w')
        for p in points:
            f.write("v {:3f} {:3f} {:3f}\n".format(p[0],p[1],p[2]))

        f.write("v {:3f} {:3f} {:3f}\n".format(outside[0], outside[1], outside[2]))

        nump = points.shape[0]
        for i,p in enumerate(points):
            f.write("l {} {}\n".format((i)%nump+1,(i+1)%nump+1))
            f.write("l {} {}\n".format(i+1,nump+1))

        f.close()

    def write_points(self,m,points,filename="points",count=0, color=None):

        path = os.path.join(os.path.dirname(m['planes']))
        filename = os.path.join(path,'{}.off'.format(filename))

        f = open(filename, 'w')
        f.write("OFF\n")
        f.write("{} 0 0\n".format(points.shape[0]))
        for p in points:
            f.write("{:.3f} {:.3f} {:.3f}\n".format(p[0],p[1],p[2]))
        f.close()

    def write_surface_to_off(self,filename,points,facets):

        f = open(filename[:-3]+"off",'w')
        f.write("COFF\n")
        f.write("{} {} 0\n".format(points.shape[0],len(facets)))
        for p in points:
            f.write("{} {} {}\n".format(p[0],p[1],p[2]))
        for i,face in enumerate(facets):
            f.write("{}".format(len(face)))
            for v in face:
                f.write(" {}".format(v))
            f.write('\n')
        f.close()

    def write_colored_surface_to_ply(self, filename, points, facets, colors=None):

        col = colors if colors is not None else (np.random.random(size=3) * 255).astype(int)

        f = open(filename, 'w')

        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(len(points)))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("element face {}\n".format(len(facets)))
        f.write("property list uchar int vertex_index\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for v in points:
            f.write("{} {} {}\n".format(v[0], v[1], v[2]))
        for i,fa in enumerate(facets):
            f.write("{}".format(len(fa)))
            for v in fa:
                f.write(" {}".format(v))
            for c in col[i]:
                f.write(" {}".format(c))
            f.write("\n")



