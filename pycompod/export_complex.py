import os
import numpy as np
import logging
from .logger import make_logger

class PolyhedralComplexExporter:
    
    def __init__(self, logger=None, verbosity=logging.INFO):
        """
        This class is mainly for debugging the PolyhedralComplex class by allowing to export all kinds of intermediate results.

        :param verbosity: 
        """

        if logger is None:
            self.logger = make_logger(name="COMPOD_EXPORTER", level=verbosity)
        else:
            self.logger = logger

    def export_label_colored_cells(self, path, graph, cells, occs, mode="occ", type_colors=None):

        from fancycolor import GradientColor2D

        inside_weight = occs[:, 0]
        outside_weight = occs[:, 1]

        icol = GradientColor2D("Reds", inside_weight.min(), inside_weight.max()).get_rgb(inside_weight)
        ocol = GradientColor2D("Greens", outside_weight.min(), outside_weight.max()).get_rgb(outside_weight)

        for i, node in enumerate(graph.nodes):

            # if graph.nodes[node].get("bounding_box", 0):
            #     continue

            st = "out" if inside_weight[i] <= outside_weight[i] else "in"
            if mode == "occ":
                color = np.array(ocol[i])[:3] if inside_weight[i] <= outside_weight[i] else np.array(icol[i][:3])
                self.write_cell(os.path.join(path, "labelling_cells_{}".format(st)),
                                                cells[node],
                                                count=str(node) + st, color=color)
            elif mode == "type":
                color = type_colors[i]
                self.write_cell(
                    os.path.join(path, "labelling_cells_type_{}".format(st)), cells[node],
                    count=str(node) + st, color=color)
            else:
                self.logger.error("not a valid type for labelled cell debug export")

            # self.complexExporter.write_points(os.path.join(path,"labelling_points"),color=color, count=str(node)+st,
            #                                       points=np.concatenate(cell_points),normals=np.concatenate(cell_normals))

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




    def write_cell(self,path,polyhedron,points=None,normals=None,filename=None,count=0,color=None,inside_vert_count=0):
        """
        :param polyhedron: A sage polyhedron
        :param points: A point set
        :param normals: A normal set
        :param filename:
        :param subfolder:
        :param count:
        :param color:
        :param inside_vert_count:
        """
        c = color if color is not None else np.random.randint(0,255,size=3)

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
        if normals is not None:
            for v in verts:
                f.write(v + " 0 0 0 {} {} {}\n".format(c[0],c[1],c[2]))
        else:
            for v in verts:
                f.write(v + " {} {} {}\n".format(c[0],c[1],c[2]))
        faces = ss[3]
        for fa in faces:
            f.write(fa[0] + " ")
            for ffa in fa[2:].split(' '):
                f.write(str(int(ffa)+inside_vert_count)+" ")
            f.write("\n")


        if points is not None and normals is not None:
            for i,p in enumerate(points):
                n = normals[i]
                f.write("v {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {} {} {}\n".format(p[0],p[1],p[2],n[0],n[1],n[2],c[0],c[1],c[2]))
        elif points is not None and normals is None:
            for i,p in enumerate(points):
                f.write("v {:.3f} {:.3f} {:.3f} {} {} {}\n".format(p[0],p[1],p[2],c[0],c[1],c[2]))

        f.close()


    def write_facet(self,path,facet,count=0, color=None):

        c = color if color is not None else np.random.randint(0,255,size=3)

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


    # def write_graph_edge(self,m,graph,e0,e1):
    # 
    #     assert (len(graph[e0][e1]["vertices"]) > 2)
    # 
    #     pts = []
    #     for v in graph[e0][e1]["vertices"]:
    #         pts.append(tuple(v))
    #     pts = list(set(pts))
    #     intersection_points = np.array(pts, dtype=object)
    # 
    #     correct_order = self.cellComplex._sort_vertex_indices_by_angle(intersection_points.astype(float),
    #                                                  graph[e0][e1]["supporting_plane"])
    #     assert (len(intersection_points) == len(correct_order))
    #     intersection_points = intersection_points[correct_order]
    # 
    #     if (len(intersection_points) < 3):
    #         self.logger.warn("Graph edge with less than three polygon vertices")
    #         return
    # 
    #     ## orient triangle
    # 
    #     ## TODO: problem here is that orientation doesn't work when points are on the same line, because then e1 and e2 are coplanar
    #     outside = graph.nodes[e0]["convex"].centroid()
    #     ei1 = (intersection_points[1] - intersection_points[0]).astype(float)
    #     ei1 = ei1 / np.linalg.norm(ei1)
    #     ei2 = (intersection_points[-1] - intersection_points[0]).astype(float)
    #     ei2 = ei2 / np.linalg.norm(ei2)
    #     # e2 = e1
    #     # s=1
    #     # while np.isclose(np.arccos(np.dot(e1,e2)),0,rtol=1e-02):
    #     #     s+=1
    #     #     e2 = (intersection_points[s] - intersection_points[0]).astype(float)
    #     #     e2 = e2/np.linalg.norm(e2)
    #     ei3 = (outside - intersection_points[0]).astype(float)
    #     ei3 = ei3 / np.linalg.norm(ei3)
    #     if self.cellComplex._orient_triangle(ei1, ei2, ei3):
    #         intersection_points = np.flip(intersection_points, axis=0)
    # 
    #     id = graph[e0][e1]["id"]
    #     filename = os.path.join(os.path.dirname(m["planes"]),"graph_facets",str(id)+".off")
    #     os.makedirs(os.path.dirname(filename), exist_ok=True)
    #     self.write_off(filename, points=intersection_points.astype(float),facets=[np.arange(len(intersection_points))],color=graph[e0][e1]["color"])

    def write_facet_with_outside_centroid(self, m, points, outside, count=0):

        filename = os.path.join(os.path.dirname(m["planes"]),"facets_with_outside_centroid",str(count)+".obj")
        os.makedirs(os.path.dirname(filename),exist_ok=True)

        f = open(filename, 'w')
        for p in points:
            f.write("v {:6f} {:6f} {:6f}\n".format(p[0],p[1],p[2]))

        f.write("v {:6f} {:6f} {:6f}\n".format(outside[0], outside[1], outside[2]))

        nump = points.shape[0]
        for i,p in enumerate(points):
            f.write("l {} {}\n".format((i)%nump+1,(i+1)%nump+1))
            f.write("l {} {}\n".format(i+1,nump+1))

        f.close()

    def write_points(self,path,points,normals=None,count=0, color=None):

        c = color if color is not None else np.random.randint(0,255,size=3)

        os.makedirs(path,exist_ok=True)
        filename = os.path.join(path,str(count)+'.off')

        f = open(filename, 'w')
        if normals is None:
            f.write("COFF\n")
        else:
            f.write("CNOFF\n")
        f.write("{} 0 0\n".format(points.shape[0]))
        if normals is None:
            for p in points:
                f.write("{:.3f} {:.3f} {:.3f} {} {} {}\n".format(p[0],p[1],p[2],c[0],c[1],c[2]))
        else:
            assert len(points) == len(normals)
            for i,p in enumerate(points):
                n=normals[i]
                f.write("{:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {} {} {}\n".format(p[0], p[1], p[2], n[0], n[1], n[2], c[0], c[1], c[2]))
        f.close()


    def write_surface(self,filename,points,facets,**kwargs):

        file_type = os.path.splitext(filename)[1]
        if(file_type == ".ply"):
            self.write_surface_to_ply(filename,points,facets,**kwargs)
        elif(file_type == ".off"):
            self.write_surface_to_off(filename,points,facets,**kwargs)
        elif(file_type == ".obj"):
            self.write_surface_to_obj(filename,points,facets,**kwargs)
        else:
            raise NotImplementedError
            self.logger.error("{} is not a valid file type for surface export.".format(file_type))
            return 1

    def write_surface_to_off(self,filename,points,facets,pcolors=[]):

        f = open(filename,'w')

        if len(pcolors):
            f.write("COFF\n")
        else:
            f.write("OFF\n")

        f.write("{} {} 0\n".format(points.shape[0],len(facets)))
        for i,p in enumerate(points):
            f.write("{:6f} {:6f} {:6f}".format(p[0], p[1], p[2]))
            if len(pcolors):
                c = pcolor[i]
                f.write(" {} {} {}".format(c[0], c[1], c[2]))
            f.write("\n")

        for i,face in enumerate(facets):
            f.write("{}".format(len(face)))
            for v in face:
                f.write(" {}".format(v))
            f.write('\n')
        f.close()

    def write_surface_to_obj(self,filename,points,facets,pcolors=[]):

        f = open(filename,'w')

        for i,p in enumerate(points):
            f.write("v {:6f} {:6f} {:6f}".format(p[0], p[1], p[2]))
            # f.write("v {} {} {}".format(p[0], p[1], p[2]))
            if len(pcolors):
                c = pcolors[i]
                f.write(" {} {} {}".format(int(c[0]), int(c[1]), int(c[2])))
            f.write("\n")

        for i,face in enumerate(facets):
            f.write("f")
            for v in face:
                f.write(" {}".format(v+1))
            f.write('\n')
        f.close()


    def write_surface_to_ply(self,filename,points,facets,pnormals=[],pcolors=[],fcolors=[]):

        # TODO: write this function so I can export point normals

        f = open(filename,'w')

        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(len(points)))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if len(pnormals):
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
        if len(pcolors):
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("element face {}\n".format(len(facets)))
        f.write("property list uchar int vertex_indices\n")
        if len(fcolors):
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        for i,v in enumerate(points):
            f.write("{:6f} {:6f} {:6f}".format(v[0], v[1], v[2]))
            if len(pnormals):
                n = pnormals[i]
                f.write(" {:6f} {:6f} {:6f}".format(n[0], n[1], n[2]))
            if len(pcolors):
                c = pcolors[i]
                f.write(" {} {} {}".format(c[0], c[1], c[2]))
            f.write("\n")

        for i,fa in enumerate(facets):
            f.write("{}".format(len(fa)))
            for v in fa:
                f.write(" {}".format(v))
            if len(fcolors):
                c = fcolors[i]
                f.write(" {} {} {}".format(c[0],c[1],c[2]))
            f.write("\n")

        f.close()



    def write_colored_soup_to_ply(self, filename, points, facets, pcolors=None, fcolors=None):

        # col = colors if colors is not None else (np.random.random(size=3) * 255).astype(int)

        f = open(filename, 'w')

        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(len(points)))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("element face {}\n".format(len(facets)))
        f.write("property list uchar int vertex_indices\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i,v in enumerate(points):
            f.write("{:6f} {:6f} {:6f} {} {} {}\n".format(v[0], v[1], v[2], pcolors[i][0],pcolors[i][1],pcolors[i][2]))
        for i,fa in enumerate(facets):
            f.write("{}".format(len(fa)))
            for v in fa:
                f.write(" {}".format(v))
            for c in fcolors[i]:
                f.write(" {}".format(c))
            f.write("\n")
    
    
       


