#pragma once

#include <random>

#include <cgal_typedefs.h>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>

using namespace std;

namespace nb = nanobind;
using namespace nb::literals;


#include <CGAL/Surface_mesh.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel EPICK;
typedef CGAL::Surface_mesh<EPICK::Point_3>   Inexact_Mesh;


#include <CGAL/Side_of_triangle_mesh.h>
#include <CGAL/point_generators_3.h>
#include <CGAL/Random.h>
typedef CGAL::Side_of_triangle_mesh<Inexact_Mesh, EPICK> Point_inside;
typedef CGAL::AABB_face_graph_triangle_primitive<Inexact_Mesh> Primitive;
typedef CGAL::AABB_traits<EPICK, Primitive> AABB_Traits;
typedef CGAL::AABB_tree<AABB_Traits> AABB_Tree;
typedef AABB_Tree::Point_and_primitive_id Point_and_primitive_id;
typedef boost::optional<AABB_Tree::Intersection_and_primitive_id<EPICK::Ray_3>::Type> Ray_intersection;

#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Delaunay_triangulation_cell_base_3.h>
#include <CGAL/Triangulation_vertex_base_3.h>
typedef CGAL::Triangulation_vertex_base_3<EPICK>    VB;
typedef CGAL::Triangulation_cell_base_3<EPICK>        CB;         // cell base

// Delaunay triangulation data structure
typedef CGAL::Triangulation_data_structure_3<VB, CB>                Tds;        // triangulation data structure
typedef CGAL::Delaunay_triangulation_3<EPICK, Tds>                  Delaunay;   // delaunay triangulation based on triangulation data structure



class pyPDL
{
public:
//    std::shared_ptr<spdlog::logger> _logger;
    pyPDL(int);
    ~pyPDL();
    int load_mesh(const string filename);
    const float _label(vector<EPICK::Point_3>&);
    vector<double>
    label_cells(const nb::ndarray<int, nb::shape<nb::any>>& points_len, const nb::ndarray<double, nb::shape<nb::any, 3>>& points);
    void export_points(const string filename);
private:
    void _init_tree();
    // init random generator for coloring
    default_random_engine _generator;

    int _n_test_points;
    vector<std::tuple<EPICK::Point_3, CGAL::Color>> _all_sampled_points;
    Inexact_Mesh _gt_mesh;
    AABB_Tree* _tree;

};


