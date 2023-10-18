//#pragma once


//#include <cgal_typedefs.h>
//#include <boost_typedefs.h>

//#include <cdt.h>
//#include <mesh.h>
//#include <random>
//#include <nanobind/ndarray.h>
//#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
//#include <CGAL/Surface_mesh.h>
//#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>

//using namespace std;
//namespace nb = nanobind;

//template <typename Kernel>
//class pyPDSE
//{
//public:

//    typedef typename Kernel::Point_3 Point;
//    typedef typename Kernel::Point_2 Point2;
//    typedef typename CGAL::Surface_mesh<Point> Mesh;

//    pyPDSE(int verbosity = 1, bool debug_export = false);

//    int _verbosity;
//    bool _debug_export;

////    void load_soup(const nb::ndarray<double, nb::shape<nb::any, 3>>& points,
////                   const nb::ndarray<int, nb::shape<nb::any>>& polygons,
////                   const nb::ndarray<int, nb::shape<nb::any>>& regions);

////    int soup_to_mesh(const int triagulate);
////    int remesh_planar_regions();

//    vector<vector<int>> get_cycles(const nb::ndarray<int, nb::shape<nb::any, 2>>& edges);

//    pair<vector<vector<int>>,bool> get_cdt_of_regions_with_holes(nb::ndarray<double, nb::shape<nb::any, 2>>& points, vector<vector<int>>& cycles);

//    int load_soup(const nb::ndarray<double, nb::shape<nb::any, 3>>& points, const nb::ndarray<int, nb::shape<nb::any>>& polygons);
//    int load_triangle_soup(const nb::ndarray<double, nb::shape<nb::any, 3>>& points, const nb::ndarray<int, nb::shape<nb::any,3>>& triangles);
//    int soup_to_mesh(const bool triangulate, const bool stitch_borders);
//    int save_mesh(const string filename);

//    // validity testing
//    int is_mesh_intersection_free(const string filename);
//    int is_mesh_watertight(const string filename);

//    SMesh<Kernel> _smesh;

//    typedef typename SMesh_CDT<Kernel>::CDT CDT;

//};

//struct GraphCycles{

//    GraphCycles(vector<vector<int>>& cycles);

//    template < typename Path, typename Graph >
//    void cycle(const Path& p, const Graph& g);

//    vector<vector<int>>& _cycles;

//};

