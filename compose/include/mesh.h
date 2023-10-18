#pragma once

using namespace std;

#include <cgal_typedefs.h>
#include <boost_typedefs.h>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/directed_graph.hpp>
#include <boost/graph/tiernan_all_cycles.hpp>

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#include <CGAL/Surface_mesh.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <CGAL/Polygon_mesh_processing/stitch_borders.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>
#include <CGAL/Polygon_mesh_processing/repair_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Polygon_mesh_processing/distance.h>
#include <boost/property_map/property_map.hpp>
#include <CGAL/boost/graph/properties.h>
#include <CGAL/boost/graph/iterator.h>

#include <plane.h>


namespace PMP = CGAL::Polygon_mesh_processing;


//typedef boost::graph_traits<Mesh>::Mesh::Face_index Mesh::Face_index;
//typedef boost::graph_traits<Mesh>::vertex_descriptor vertex_descriptor;
//typedef boost::graph_traits<Mesh>::edge_descriptor edge_descriptor;
//typedef boost::graph_traits<Mesh>::halfedge_descriptor halfedge_descriptor;

template <typename Kernel>
class SMesh{
public:

    typedef typename Kernel::Point_3 Point;
    typedef typename Kernel::Point_2 Point2;
    typedef typename CGAL::Surface_mesh<Point> Mesh;

    SMesh(int verbosity = 1, bool debug_export = false);
    ~SMesh(){spdlog::drop("SMesh");}

    int load_soup_from_npz(const string filename);
    int save_mesh(const string filename);
    int save_mesh(const string filename, Mesh& mesh);
    void _save_region_mesh(const vector<typename Mesh::Face_index>& region, const string name);
    int soup_to_mesh(const bool triangulate=false, const bool stitch_borders=true);
    int soup_to_mesh_no_repair();

    int remesh_planar_regions(const bool triangulate = false, const bool simplify_edges = true);
    int remesh_almost_planar_patches(const int triangulate);
    int remesh_planar_patches(const int triangulate);

    void _color_mesh_by_region();
    void _color_mesh_boundary(Mesh& mesh, int seed);

    int _merge_region_meshes(const vector<Mesh>& meshes);
    void _get_corner_vertices();

    bool _debug_export;
    shared_ptr<spdlog::logger> _logger;

    Mesh _mesh;
    Mesh _simplified_mesh;
    vector<Point> _points;
    vector<Polygon> _polygons;
    vector<Plane<Kernel>> _planes;
    vector<CGAL::Color> _colors;

    vector<int> _polygon_to_region;
    map<int, vector<Polygon>> _region_to_polygons;

    map<int, typename Mesh::Face_index> _polygon_to_face;
    map<int, typename Mesh::Vertex_index> _point_to_vertex;
    typename Mesh::Property_map<typename Mesh::Vertex_index, bool> _vertex_is_corner;
    typename Mesh::Property_map<typename Mesh::Vertex_index, CGAL::Color> _vcolor;
    map<typename Mesh::Face_index, int> _face_to_region;
    map<int, vector<typename Mesh::Face_index>> _region_to_faces;

    vector<vector<typename Mesh::Vertex_index>> _cycles_simplified;
    vector<vector<typename Mesh::Vertex_index>> _cycles_full;

};


template <typename Kernel>
struct CF{

    CF(SMesh<Kernel>& SMesh);


    template < typename Path, typename Graph >
    void cycle(const Path& p, const Graph& g);

private:
    SMesh<Kernel>& _smesh;


};

