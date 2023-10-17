#include <iostream>
#include <pypdse.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>

#include <cgal_typedefs.h>
#include <boost_typedefs.h>

#include <cdt.h>

using namespace std;
namespace nb = nanobind;
using namespace nb::literals;

namespace fs = boost::filesystem;


pyPDSE::pyPDSE(int verbosity, bool debug_export){
    // init a PDS
    _verbosity = verbosity;
    _debug_export = debug_export;
    _PDSE = PDSE(verbosity);

}

//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// SOUP TO SIMPLIFIED MESH /////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////

GraphCycles::GraphCycles(vector<vector<int>>& cycles) : _cycles(cycles)
{};

template < typename Path, typename Graph >
void GraphCycles::cycle(const Path& p, const Graph& g)
{
    // from here: https://stackoverflow.com/a/71630444/20795095

    vector<int> cycle;
    // Get the property map containing the vertex indices
    // so we can print them.
    typedef typename boost::property_map< Graph, boost::vertex_index_t >::const_type IndexMap;
    IndexMap indices = boost::get(boost::vertex_index, g);

    // Iterate over path printing each vertex that forms the cycle.
    typename Path::const_iterator i, end = p.end();
    for (i = p.begin(); i != end; ++i)
    {
        auto id = boost::get(indices, *i);
//        cout << id << " ";
        cycle.push_back(id);
    }
//    cout << endl;
    cycle.push_back(cycle[0]);
    _cycles.push_back(cycle);
}


vector<vector<int>> pyPDSE::get_cycles(const nb::ndarray<int, nb::shape<nb::any, 2>>& edges){

    Graph graph;
    for(size_t i = 0; i < edges.shape(0); i++){
        boost::add_edge(edges(i,0),edges(i,1),graph);
    }

    vector<vector<int>> cycles;

    GraphCycles gc(cycles);
    boost::tiernan_all_cycles(graph,gc);

    return cycles;

}


void mark_domains(CDT& ct,
             CDT::Face_handle start,
             int index,
             std::list<CDT::Edge>& border )
{
  if(start->info().nesting_level != -1){
    return;
  }
  std::list<CDT::Face_handle> queue;
  queue.push_back(start);
  while(! queue.empty()){
    CDT::Face_handle fh = queue.front();
    queue.pop_front();
    if(fh->info().nesting_level == -1){
      fh->info().nesting_level = index;
      for(int i = 0; i < 3; i++){
        CDT::Edge e(fh,i);
        CDT::Face_handle n = fh->neighbor(i);
        if(n->info().nesting_level == -1){
          if(ct.is_constrained(e)) border.push_back(e);
          else queue.push_back(n);
        }
      }
    }
  }
}
//explore set of facets connected with non constrained edges,
//and attribute to each such set a nesting level.
//We start from facets incident to the infinite vertex, with a nesting
//level of 0. Then we recursively consider the non-explored facets incident
//to constrained edges bounding the former set and increase the nesting level by 1.
//Facets in the domain are those with an odd nesting level.
void mark_domains(CDT& cdt)
{
  for(CDT::Face_handle f : cdt.all_face_handles()){
    f->info().nesting_level = -1;
  }
  std::list<CDT::Edge> border;
  mark_domains(cdt, cdt.infinite_face(), 0, border);
  while(! border.empty()){
    CDT::Edge e = border.front();
    border.pop_front();
    CDT::Face_handle n = e.first->neighbor(e.second);
    if(n->info().nesting_level == -1){
      mark_domains(cdt, n, e.first->info().nesting_level+1, border);
    }
  }
}

pair<vector<vector<int>>,bool> pyPDSE::get_cdt_of_regions_with_holes
(nb::ndarray<double, nb::shape<nb::any, 2>>& points, vector<vector<int>>& cycles){

    // make a 2D constrained delaunay triangulation of the region
    CDT cdt;
    map<CDT::Vertex_handle,int> two_d_to_three_d;
    for(auto cycle : cycles){

        // insert points
        for(int i = 0; i < cycle.size()-1; i++){
            Point2 pts(points(cycle[i],0),points(cycle[i],1));
            Point2 ptt(points(cycle[i+1],0),points(cycle[i+1],1));
            auto dts = cdt.push_back(pts);
            two_d_to_three_d[dts] = cycle[i];
            cdt.insert_constraint(pts,ptt);
        }
    }

    // TODO: there is now an internal function in CGAL mark_domain_in_triangulation() that can take care of this.
    mark_domains(cdt);
    assert(cdt.is_valid());

    // get the triangle indices
    vector<vector<int>> region_triangles;
    CDT::Finite_faces_iterator fit;
    bool has_hole = false;
    for(fit = cdt.finite_faces_begin(); fit != cdt.finite_faces_end(); fit++){
        if( !fit->info().in_domain() ){
            if(fit->info().nesting_level > 0)
                has_hole = true;
            continue;
        }
        vector<int> triangle;
        auto face = *fit;
        for(size_t i = 0; i < 3; i++)
            triangle.push_back(two_d_to_three_d[face.vertex(i)]);
        region_triangles.push_back(triangle);
    }

    return pair(region_triangles,has_hole);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// SOUP TO UNSIMPLIFIED MESH ////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////
int pyPDSE::load_soup(const nb::ndarray<double, nb::shape<nb::any, 3>>& points,
                      const nb::ndarray<int, nb::shape<nb::any>>& polygons
                      ){

    _smesh = SMesh(_verbosity,_debug_export);


    _smesh._points.clear();
    _smesh._polygons.clear();

    for (size_t i = 0; i < points.shape(0); i++){
        _smesh._points.push_back(Point(points(i,0),points(i,1),points(i,2)));
    }


    int n = 0;
    for (size_t i = 0; i < polygons.shape(0); i++){
        Polygon poly;
        for(size_t j = 0; j < polygons(i); j++){
            poly.push_back(j+n);
        }
        n+=poly.size();
        _smesh._polygons.push_back(poly);
    }
    return 0;
}

int pyPDSE::load_triangle_soup(const nb::ndarray<double, nb::shape<nb::any, 3>>& points,
                      const nb::ndarray<int, nb::shape<nb::any,3>>& triangles
                      ){

    _smesh = SMesh(_verbosity,_debug_export);

    _smesh._points.clear();
    _smesh._polygons.clear();

    for (size_t i = 0; i < points.shape(0); i++){
        _smesh._points.push_back(Point(points(i,0),points(i,1),points(i,2)));
    }

    for (size_t i = 0; i < triangles.shape(0); i++){
        Polygon poly;
        poly.push_back(triangles(i,0));
        poly.push_back(triangles(i,1));
        poly.push_back(triangles(i,2));
        _smesh._polygons.push_back(poly);
    }
    return 0;
}

int pyPDSE::soup_to_mesh(const bool triangulate, const bool stitch_borders){

    return _smesh.soup_to_mesh(triangulate, stitch_borders);
}

int pyPDSE::save_mesh(const string filename){

    return _smesh.save_mesh(filename);
}

int pyPDSE::is_mesh_intersection_free(const string filename){

    Mesh mesh;
    if(!PMP::IO::read_polygon_mesh(filename, mesh) || !CGAL::is_triangle_mesh(mesh))
    {
      std::cerr << "Invalid input." << std::endl;
      return 0;
    }
    return PMP::does_self_intersect<CGAL::Parallel_if_available_tag>(mesh, CGAL::parameters::vertex_point_map(get(CGAL::vertex_point, mesh)));

}

int pyPDSE::is_mesh_watertight(const string filename){

    Mesh mesh;
    if(!PMP::IO::read_polygon_mesh(filename, mesh) || !CGAL::is_triangle_mesh(mesh))
    {
      std::cerr << "Invalid input." << std::endl;
      return 0;
    }
    return CGAL::is_closed(mesh);

}


//////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////// THIS STUFF IS FOR CONTROLLING THE PURE CGAL VERSION OF PDSE ///////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////
//int pyPDSE::save_mesh(const string filename){

//    return _PDSE.make_mesh(filename);

//}

//int pyPDSE::remesh_planar_regions(){

//    return _PDSE.remesh_planar_regions();
//}


NB_MODULE(libPYPDSE, m) {
    nb::class_<pyPDSE>(m, "pdse")
            .def(nb::init<int,bool>(),"verbosity"_a = 0, "debug_export"_a = false)
//            .def("load_soup", nb::overload_cast<const nb::ndarray<double, nb::shape<nb::any, 3>>&,
//                 const nb::ndarray<int, nb::shape<nb::any>>&>(&pyPDSE::load_soup), "points"_a, "polygons"_a, "Load a polygon soup.")
//            .def("load_soup", nb::overload_cast<const nb::ndarray<double, nb::shape<nb::any, 3>>&,
//                 const nb::ndarray<int, nb::shape<nb::any,3>>&>(&pyPDSE::load_soup), "points"_a, "triangles"_a, "Load a triangle soup.")
            .def("is_mesh_watertight", &pyPDSE::is_mesh_watertight, "filename"_a, "Check if a mesh is watertight.")
            .def("is_mesh_intersection_free", &pyPDSE::is_mesh_intersection_free, "filename"_a, "Check if a mesh is free of self-intersections.")
            .def("load_soup", &pyPDSE::load_soup, "points"_a, "polygons"_a, "Load a polygon soup.")
            .def("load_triangle_soup", &pyPDSE::load_triangle_soup, "points"_a, "triangles"_a, "Load a triangle soup.")
            .def("soup_to_mesh", &pyPDSE::soup_to_mesh, "triangulate"_a, "stitch_borders"_a, "Generate a polygon mesh from the polygon soup.")
            .def("save_mesh", &pyPDSE::save_mesh, "filename"_a, "Save a mesh.")
            .def("get_cycles", &pyPDSE::get_cycles, "boundary_edges"_a ,"Get closed cycles from a set of boundary edges.")
            .def("get_cdt_of_regions_with_holes", &pyPDSE::get_cdt_of_regions_with_holes, "points2d"_a, "constrained_edges"_a, "Get the Constrained Delaunay Triangulation from 2D points and edges.");

//            .def("load_soup", &pyPDSE::load_soup)
//            .def("soup_to_mesh", &pyPDSE::soup_to_mesh, "triangulate"_a ,"Load a point cloud with normals or a point group file.")
//            .def("remesh_planar_regions", &pyPDSE::remesh_planar_regions, "Load a point cloud with normals or a point group file.");
}


