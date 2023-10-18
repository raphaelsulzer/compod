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



#include <CGAL/Polygon_mesh_processing/region_growing.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup_extension.h>

using namespace std;
namespace fs = boost::filesystem;

template <typename Kernel>
SMesh<Kernel>::SMesh(int verbosity, bool debug_export){

    _debug_export = debug_export;


    if(spdlog::get("SMesh")){
        _logger = spdlog::get("SMesh");
    }
    else{
        _logger = spdlog::stdout_color_mt("SMesh");
    }

    if(verbosity == 0)
        _logger->set_level(spdlog::level::warn);
    else if(verbosity == 1)
        _logger->set_level(spdlog::level::info);
    else if(verbosity == 2)
        _logger->set_level(spdlog::level::debug);
    else
        _logger->set_level(spdlog::level::off);

    spdlog::set_pattern("[%H:%M:%S] [%n] [%l] %v");

}


template <typename Kernel>
int SMesh<Kernel>::save_mesh(const string filename, Mesh& outmesh){

    auto path = fs::path(filename);

    if(!fs::is_directory(path.parent_path()))
        fs::create_directories(path.parent_path());

    if(outmesh.number_of_faces()>0){
        _logger->debug("Save surface mesh to {}",filename);
    }
    else if(_mesh.number_of_faces()>0){
        outmesh = _mesh;
        _logger->debug("Save surface mesh to {}",filename);
    }
    else{
        _logger->error("No mesh available to save. First run soup_to_mesh().");
        return 1;
    }

    CGAL::IO::write_polygon_mesh(filename,outmesh);
    return 0;

}

template <typename Kernel>
int SMesh<Kernel>::save_mesh(const string filename){

    Mesh mesh;
    return save_mesh(filename,mesh);

}

template <typename Kernel>
int SMesh<Kernel>::_merge_region_meshes(const vector<Mesh>& meshes){

    vector<Point> new_points;
    vector<Polygon> new_polys;
    int n = 0;
    for(auto mesh : meshes){

        for(auto p : mesh.points()){
            new_points.push_back(p);
        }

        for(auto fi : mesh.faces()){

            Polygon poly;
            CGAL::Vertex_around_face_circulator<Mesh> vcirc(mesh.halfedge(fi), mesh), done(vcirc);
            do{
                poly.push_back(*vcirc++ + n);
            }while (vcirc != done);
            new_polys.push_back(poly);
        }
        n+=mesh.number_of_vertices();

    }

    _polygons = new_polys;
    _points = new_points;


    return 0;

}

#include <xtensor-io/xnpz.hpp>
#include <xtensor/xnpy.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xtensor.hpp>
template <typename Kernel>
int SMesh<Kernel>::load_soup_from_npz(const string filename){

    /////// with xtensor
    auto arr = xt::load_npz(filename);

    if(arr.find("points") == arr.end()){
        _logger->error("No points array found in {}", filename);
        return 1;
    }
    if(arr.find("polygons") == arr.end()){
        _logger->error("No polygons array found in {}", filename);
        return 1;
    }


    auto pts = arr["points"].cast<double>();
    auto polys = arr["polygons"].cast<int>();
    auto polygon_regions = arr["polygon_regions"].cast<int>();
    auto planes = arr["planes"].cast<float>();
    auto colors = arr["colors"].cast<int>();

    _points.clear();
    _polygons.clear();
    _colors.clear();
    _polygon_to_region.clear();
    _region_to_polygons.clear();
    _planes.clear();


    for (size_t i = 0; i < pts.shape(0); i++){
        _points.push_back(Point(pts(i,0),pts(i,1),pts(i,2)));
    }


    int n = 0;
    for (size_t i = 0; i < polys.shape(0); i++){
        Polygon poly;
        for(size_t j = 0; j < polys(i); j++){
            poly.push_back(j+n);
        }
        n+=poly.size();
        _region_to_polygons[polygon_regions[i]].push_back(poly);
        _polygons.push_back(poly);
    }


    for (size_t i = 0; i < planes.shape(0); i++){
        _planes.push_back(Plane({planes(i,0),planes(i,1),planes(i,2),planes(i,3)}));
    }

    for (size_t i = 0; i < colors.shape(0); i++){
        _colors.push_back(CGAL::Color(colors(i,0),colors(i,1),colors(i,2)));
    }

    for (size_t i = 0; i < polygon_regions.shape(0); i++){
        _polygon_to_region.push_back(polygon_regions(i));
    }

    int min = *min_element(_polygon_to_region.begin(), _polygon_to_region.end());
    int max = *max_element(_polygon_to_region.begin(), _polygon_to_region.end());
    assert(min >= 0);
    assert(max < _planes.size());

    return 0;


}




template <typename Kernel>
int SMesh<Kernel>::soup_to_mesh(const bool triangulate, const bool stitch_borders){

    _mesh.clear();

//// These 3 are boundled in repair_polygon_soup
//    PMP::remove_isolated_points_in_polygon_soup(_points,_polygons);
    PMP::merge_duplicate_points_in_polygon_soup(_points, _polygons);
    PMP::merge_duplicate_polygons_in_polygon_soup(_points, _polygons);

    PMP::repair_polygon_soup(_points, _polygons);
    PMP::orient_polygon_soup(_points, _polygons);

//    PMP::duplicate_non_manifold_edges_in_polygon_soup(_points,_polygons);
    PMP::polygon_soup_to_polygon_mesh(_points, _polygons, _mesh);


    if(stitch_borders){
        _logger->debug("Stitch borders...");
        PMP::stitch_borders(_mesh);
    }

    vector<typename Mesh::Halfedge_index> boundaries;
    PMP::extract_boundary_cycles(_mesh,back_inserter(boundaries));
    for(auto boundary : boundaries){
        PMP::triangulate_hole(_mesh,boundary,PMP::parameters::use_2d_constrained_delaunay_triangulation(false));
    }


    if(triangulate){
        _logger->debug("Triangulate...");

        // this function calls triangulate_hole_polyline(), which is why sometimes
        // mesh with holes is closed after triangulation
        PMP::triangulate_faces(_mesh);

        if(CGAL::is_closed(_mesh)){
                PMP::orient_to_bound_a_volume(_mesh);
        }
        else{
            _logger->warn("Mesh is not closed!");
        }
        if (!PMP::is_outward_oriented(_mesh)) {
            PMP::reverse_face_orientations(_mesh);
        }
    }

    // see here how to get a map from polygon faces to triangle faces:
    // https://doc.cgal.org/latest/Polygon_mesh_processing/Polygon_mesh_processing_2triangulate_faces_split_visitor_example_8cpp-example.html

    return 0;

}





