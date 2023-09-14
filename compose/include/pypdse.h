#pragma once


#include <cgal_typedefs.h>
#include <boost_typedefs.h>

#include <pdse.h>
#include <mesh.h>
#include <random>
#include <nanobind/ndarray.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>


using namespace std;
namespace nb = nanobind;


class pyPDSE
{
public:

    pyPDSE(int verbosity = 1, bool debug_export = false);

    int _verbosity;
    bool _debug_export;

//    void load_soup(const nb::ndarray<double, nb::shape<nb::any, 3>>& points,
//                   const nb::ndarray<int, nb::shape<nb::any>>& polygons,
//                   const nb::ndarray<int, nb::shape<nb::any>>& regions);

//    int soup_to_mesh(const int triagulate);
//    int remesh_planar_regions();

    vector<vector<int>> get_cycles(const nb::ndarray<int, nb::shape<nb::any, 2>>& edges);

    pair<vector<vector<int>>,bool> get_cdt_of_regions_with_holes(nb::ndarray<double, nb::shape<nb::any, 2>>& points, vector<vector<int>>& cycles);

    int load_soup(const nb::ndarray<double, nb::shape<nb::any, 3>>& points, const nb::ndarray<int, nb::shape<nb::any>>& polygons);
    int soup_to_mesh(const bool triangulate, const bool stitch_borders);
    int save_mesh(const string filename);


    PDSE _PDSE;

    SMesh _smesh;


};

struct GraphCycles{

    GraphCycles(vector<vector<int>>& cycles);


    template < typename Path, typename Graph >
    void cycle(const Path& p, const Graph& g);

    vector<vector<int>>& _cycles;


};

