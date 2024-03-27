#pragma once


#include <cgal_typedefs.h>
#include <boost_typedefs.h>

#include <mesh.h>
#include <random>
#include <nanobind/ndarray.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>

using namespace std;
namespace nb = nanobind;

struct FaceInfo2
{
  FaceInfo2(){}
  int nesting_level = 0;
  bool in_domain(){
    return nesting_level%2 == 1;
  }
};

template <typename Kernel>
class pyPDSE
{
public:

    typedef SMesh<Kernel> pySMesh;

    pySMesh _smesh;


    typedef typename Kernel::Point_3 Point;
    typedef typename Kernel::Point_2 Point2;
    typedef typename CGAL::Surface_mesh<Point> Mesh;

    pyPDSE(int verbosity = 1, bool debug_export = false);

    int _verbosity;
    bool _debug_export;

    vector<vector<int>> get_cycles(const nb::ndarray<int, nb::shape<nb::any, 2>>& edges);

    pair<vector<vector<int>>,bool> get_cdt_of_regions_with_holes(nb::ndarray<double, nb::shape<nb::any, 2>>& points, vector<vector<int>>& cycles);

    int load_soup(const nb::ndarray<double, nb::shape<nb::any, 3>>& points, const nb::ndarray<int, nb::shape<nb::any>>& polygons);
    int triangulate_polygon_mesh(const string filename, const string outfilename,
                                 const bool force_rebuild, const int precision);
    int load_triangle_soup(const nb::ndarray<double, nb::shape<nb::any, 3>>& points, const nb::ndarray<int, nb::shape<nb::any,3>>& triangles);
    int soup_to_mesh(const bool triangulate, const bool stitch_borders);
    int save_mesh(const string filename);

    // validity testing
    int is_mesh_intersection_free(const string filename);
    int is_mesh_watertight(const string filename);

    // // compute planar regions of a mesh
    // int compute_planar_regions(const string filename);


    // define the CDT
    typedef CGAL::Triangulation_vertex_base_2<Kernel>                       Vb;
    typedef CGAL::Triangulation_face_base_with_info_2<FaceInfo2,Kernel>     Fbb;
    typedef CGAL::Constrained_triangulation_face_base_2<Kernel,Fbb>         Fb;
    typedef CGAL::Triangulation_data_structure_2<Vb,Fb>                     TDS;


    typedef CGAL::No_constraint_intersection_tag                            NItag;
    typedef CGAL::No_constraint_intersection_requiring_constructions_tag    Itag;
    typedef CGAL::Exact_predicates_tag                                      Etag;
    using Tag = typename std::conditional<std::is_same<Kernel, EPECK>::value, Etag, Itag>::type;
    typedef CGAL::Constrained_Delaunay_triangulation_2<Kernel, TDS, Tag>    CDT;



};

struct GraphCycles{

    GraphCycles(vector<vector<int>>& cycles);

    template < typename Path, typename Graph >
    void cycle(const Path& p, const Graph& g);

    vector<vector<int>>& _cycles;

};

