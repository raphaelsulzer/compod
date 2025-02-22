#include <iostream>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>

#include <pypdse.h>
#include <cgal_typedefs.h>
#include <boost_typedefs.h>
#include <mesh.h>
// #include <region_growing.h>

using namespace std;
namespace nb = nanobind;
using namespace nb::literals;

namespace fs = boost::filesystem;

template <typename Kernel>
pyPDSE<Kernel>::pyPDSE(int verbosity, bool debug_export){
    // init a PDS
    _verbosity = verbosity;
    _debug_export = debug_export;

}

////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////// SOUP TO SIMPLIFIED MESH /////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////

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

template <typename Kernel>
vector<vector<int>> pyPDSE<Kernel>::get_cycles(const nb::ndarray<int, nb::shape<-1, 2>>& edges){

    Graph graph;
    for(size_t i = 0; i < edges.shape(0); i++){
        boost::add_edge(edges(i,0),edges(i,1),graph);
    }

    vector<vector<int>> cycles;

    GraphCycles gc(cycles);
    boost::tiernan_all_cycles(graph,gc);

    return cycles;

}

template <typename Kernel>
void mark_domains(typename pyPDSE<Kernel>::CDT& ct,
             typename pyPDSE<Kernel>::CDT::Face_handle start,
             int index,
             std::list<typename pyPDSE<Kernel>::CDT::Edge>& border )
{
  if(start->info().nesting_level != -1){
    return;
  }

  std::list<typename pyPDSE<Kernel>::CDT::Face_handle> queue;
  queue.push_back(start);
  while(! queue.empty()){
    typename pyPDSE<Kernel>::CDT::Face_handle fh = queue.front();
    queue.pop_front();
    if(fh->info().nesting_level == -1){
      fh->info().nesting_level = index;
      for(int i = 0; i < 3; i++){
        typename pyPDSE<Kernel>::CDT::Edge e(fh,i);
        typename pyPDSE<Kernel>::CDT::Face_handle n = fh->neighbor(i);
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
template <typename Kernel>
void mark_domains(typename pyPDSE<Kernel>::CDT& cdt)
{
  for(typename pyPDSE<Kernel>::CDT::Face_handle f : cdt.all_face_handles()){
    f->info().nesting_level = -1;
  }
  std::list<typename pyPDSE<Kernel>::CDT::Edge> border;
  mark_domains<Kernel>(cdt, cdt.infinite_face(), 0, border);
  while(! border.empty()){
    typename pyPDSE<Kernel>::CDT::Edge e = border.front();
    border.pop_front();
    typename pyPDSE<Kernel>::CDT::Face_handle n = e.first->neighbor(e.second);
    if(n->info().nesting_level == -1){
      mark_domains<Kernel>(cdt, n, e.first->info().nesting_level+1, border);
    }
  }
}

template <typename Kernel>
pair<vector<vector<int>>,bool> pyPDSE<Kernel>::get_cdt_of_regions_with_holes
(nb::ndarray<double, nb::shape<-1, 2>>& points, vector<vector<int>>& cycles){

    // make a 2D constrained delaunay triangulation of the region
    typename pyPDSE<Kernel>::CDT cdt;
    map<typename pyPDSE<Kernel>::CDT::Vertex_handle,int> two_d_to_three_d;
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
    mark_domains<Kernel>(cdt);
    assert(cdt.is_valid());

    // get the triangle indices
    vector<vector<int>> region_triangles;
    typename pyPDSE<Kernel>::CDT::Finite_faces_iterator fit;
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
template <typename Kernel>
int pyPDSE<Kernel>::load_polygon_soup(const nb::ndarray<double, nb::shape<-1, 3>>& points,
                      const nb::ndarray<int, nb::shape<-1>>& polygons,
                                      const nb::ndarray<int, nb::shape<-1>>& polygon_lens
                      ){

    _smesh = pyPDSE<Kernel>::pySMesh(_verbosity,_debug_export);


    _smesh._points.clear();
    _smesh._polygons.clear();

    for (size_t i = 0; i < points.shape(0); i++){
        _smesh._points.push_back(Point(points(i,0),points(i,1),points(i,2)));
    }

    int n = 0;
    int this_len;
    Polygon poly;
    for (size_t i = 0; i < polygon_lens.shape(0); i++){
        this_len = polygon_lens(i);
        poly.clear();
        for(size_t j = n; j < n+this_len; j++){
            poly.push_back(polygons(j));
        }
        n+=this_len;
        _smesh._polygons.push_back(poly);
    }

    return 0;
}


template <typename Kernel>
int pyPDSE<Kernel>::triangulate_polygon_mesh(const string filename, const string outfilename,
                                             const bool force_rebuild, const int precision){

    _smesh = pyPDSE<Kernel>::pySMesh(_verbosity,_debug_export);

    _smesh._mesh.clear();

    if(!force_rebuild)
        CGAL::IO::read_polygon_mesh(filename, _smesh._mesh);

    if(_smesh._mesh.number_of_faces() == 0){
        cout << "ERROR: " << filename << " has no faces" << endl;
        cout << "Will try to read a soup and make a mesh out of it" << endl;
    }
    else{
        if(!CGAL::is_triangle_mesh(_smesh._mesh))
            CGAL::Polygon_mesh_processing::triangulate_faces(_smesh._mesh);

        cout << "Mesh loading worked. Will export the mesh as " << outfilename << endl;
        CGAL::IO::write_polygon_mesh(outfilename,_smesh._mesh,CGAL::parameters::stream_precision(precision));
        return 0;
    }



    _smesh._mesh.clear();
    _smesh._points.clear();
    _smesh._polygons.clear();

    CGAL::IO::read_polygon_soup(filename, _smesh._points, _smesh._polygons);
    CGAL::Polygon_mesh_processing::polygon_soup_to_polygon_mesh(_smesh._points,_smesh._polygons,_smesh._mesh);

    if(_smesh._mesh.number_of_faces() == 0){
        cout << "ERROR: " << filename << " has no faces. Cannot do anything." << endl;
        return 1;
    }
    else{
        if(!CGAL::is_triangle_mesh(_smesh._mesh))
            CGAL::Polygon_mesh_processing::triangulate_faces(_smesh._mesh);

        cout << "Soup loading worked. Will export the mesh as " << outfilename << endl;
        CGAL::IO::write_polygon_mesh(outfilename,_smesh._mesh,CGAL::parameters::stream_precision(precision));
        return 0;
    }
}



template <typename Kernel>
vector<bool>
pyPDSE<Kernel>::check_mesh_contains(const nb::ndarray<double, nb::shape<-1, 3>>& points){


    Tree tree(faces(_smesh._mesh).first, faces(_smesh._mesh).second, _smesh);
    tree.accelerate_distance_queries();
    const Point_inside inside_tester(tree);

    typename Kernel::Point_3 tpoint;
    vector<bool> occupancy;
    for(size_t i = 0; i < points.shape(0); i++){
        tpoint = Point(points(i,0),points(i,1),points(i,2));
        occupancy.push_back(inside_tester(tpoint) == CGAL::ON_BOUNDED_SIDE);
    }
    return occupancy;

}


template <typename Kernel>
int pyPDSE<Kernel>::load_triangle_mesh(const nb::ndarray<double, nb::shape<-1, 3>>& points,
                                       const nb::ndarray<int, nb::shape<-1, 2>>& edges,
                                       const nb::ndarray<int, nb::shape<-1,3>>& triangles
                                       ){

    _smesh = pyPDSE<Kernel>::pySMesh(_verbosity,_debug_export);
    _smesh._mesh.clear();

    _smesh._mesh.reserve(points.shape(0),edges.shape(0),triangles.shape(0));

    for (size_t i = 0; i < points.shape(0); i++){
        _smesh._mesh.add_vertex(Point(points(i,0),points(i,1),points(i,2)));
    }

    for (size_t i = 0; i < triangles.shape(0); i++){
        _smesh._mesh.add_face(CGAL::SM_Vertex_index(triangles(i,0)),
                              CGAL::SM_Vertex_index(triangles(i,1)),
                              CGAL::SM_Vertex_index(triangles(i,2)));
    }

    return 0;
}



template <typename Kernel>
int pyPDSE<Kernel>::load_triangle_soup(const nb::ndarray<double, nb::shape<-1, 3>>& points,
                      const nb::ndarray<int, nb::shape<-1,3>>& triangles
                      ){

    _smesh = pyPDSE<Kernel>::pySMesh(_verbosity,_debug_export);

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

template <typename Kernel>
int pyPDSE<Kernel>::soup_to_mesh(const bool triangulate, const bool stitch_borders){

    return _smesh.soup_to_mesh(triangulate, stitch_borders);
}

template <typename Kernel>
int pyPDSE<Kernel>::save_mesh(const string filename){

    return _smesh.save_mesh(filename);
}

template <typename Kernel>
int pyPDSE<Kernel>::is_mesh_intersection_free(const string filename){

    typename pyPDSE<Kernel>::Mesh mesh;
    if(!PMP::IO::read_polygon_mesh(filename, mesh) || !CGAL::is_triangle_mesh(mesh))
    {
      std::cerr << "Invalid input." << std::endl;
      return 0;
    }
    return !PMP::does_self_intersect<CGAL::Parallel_if_available_tag>(mesh, CGAL::parameters::vertex_point_map(get(CGAL::vertex_point, mesh)));

}

template <typename Kernel>
int pyPDSE<Kernel>::is_mesh_watertight(const string filename){

    typename pyPDSE<Kernel>::Mesh mesh;
    if(!PMP::IO::read_polygon_mesh(filename, mesh) || !CGAL::is_triangle_mesh(mesh))
    {
      std::cerr << "Invalid input." << std::endl;
      return 0;
    }
    return CGAL::is_closed(mesh);

}




NB_MODULE(libPYPDSE, m) {
    nb::class_<pyPDSE<EPICK>>(m, "pdse")
            .def(nb::init<int,bool>(),"verbosity"_a = 0, "debug_export"_a = false)
        .def("is_mesh_watertight", &pyPDSE<EPICK>::is_mesh_watertight, "filename"_a, "Check if a mesh is watertight.")
        // .def("compute_planar_regions", &pyPDSE<EPICK>::compute_planar_regions, "filename"_a, "Compute planar regions of a mesh.")
            .def("is_mesh_intersection_free", &pyPDSE<EPICK>::is_mesh_intersection_free, "filename"_a, "Check if a mesh is free of self-intersections.")
        .def("load_polygon_soup", &pyPDSE<EPICK>::load_polygon_soup, "points"_a, "polygons"_a, "polygon_lens"_a, "Load a polygon soup.")
        .def("triangulate_polygon_mesh", &pyPDSE<EPICK>::triangulate_polygon_mesh,
             "filename"_a, "outfilename"_a, "force_rebuild"_a, "precision"_a, "Load polygon mesh or soup, triangulate and save to file.")
        .def("load_triangle_soup", &pyPDSE<EPICK>::load_triangle_soup, "points"_a, "triangles"_a, "Load a triangle soup.")
        .def("load_triangle_mesh", &pyPDSE<EPICK>::load_triangle_mesh, "points"_a, "edges"_a, "triangles"_a, "Load a triangle mesh.")
            .def("soup_to_mesh", &pyPDSE<EPICK>::soup_to_mesh, "triangulate"_a, "stitch_borders"_a, "Generate a polygon mesh from the polygon soup.")
            .def("save_mesh", &pyPDSE<EPICK>::save_mesh, "filename"_a, "Save a mesh.")
            .def("get_cycles", &pyPDSE<EPICK>::get_cycles, "boundary_edges"_a ,"Get closed cycles from a set of boundary edges.")
            .def("get_cdt_of_regions_with_holes", &pyPDSE<EPICK>::get_cdt_of_regions_with_holes, "points2d"_a, "constrained_edges"_a,
                 "Get the Constrained Delaunay Triangulation from 2D points and edges.")

            ;

    nb::class_<pyPDSE<EPECK>>(m, "pdse_exact")
            .def(nb::init<int,bool>(),"verbosity"_a = 0, "debug_export"_a = false)
            .def("is_mesh_watertight", &pyPDSE<EPECK>::is_mesh_watertight, "filename"_a, "Check if a mesh is watertight.")
            .def("is_mesh_intersection_free", &pyPDSE<EPECK>::is_mesh_intersection_free, "filename"_a, "Check if a mesh is free of self-intersections.")
            .def("load_soup", &pyPDSE<EPECK>::load_polygon_soup, "points"_a, "polygons"_a, "polygon_lens"_a, "Load a polygon soup.")
            .def("load_triangle_soup", &pyPDSE<EPECK>::load_triangle_soup, "points"_a, "triangles"_a, "Load a triangle soup.")
            .def("soup_to_mesh", &pyPDSE<EPECK>::soup_to_mesh, "triangulate"_a, "stitch_borders"_a, "Generate a polygon mesh from the polygon soup.")
            .def("save_mesh", &pyPDSE<EPECK>::save_mesh, "filename"_a, "Save a mesh.")
            .def("get_cycles", &pyPDSE<EPECK>::get_cycles, "boundary_edges"_a ,"Get closed cycles from a set of boundary edges.")
            .def("get_cdt_of_regions_with_holes", &pyPDSE<EPECK>::get_cdt_of_regions_with_holes, "points2d"_a, "constrained_edges"_a,
                 "Get the Constrained Delaunay Triangulation from 2D points and edges.")
            ;
}
