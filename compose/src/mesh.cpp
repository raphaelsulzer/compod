#include <iostream>
#include <mesh.h>
#include <cdt.h>

#include <boost_typedefs.h>

#include <CGAL/Polygon_mesh_processing/region_growing.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup_extension.h>

using namespace std;
namespace fs = boost::filesystem;


SMesh::SMesh(int verbosity, bool debug_export){

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



int SMesh::save_mesh(const string filename, Mesh& outmesh){

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

////    auto fcolor = outmesh.property_map<Mesh::Face_index, CGAL::Color>("f:color").first;
//    auto vcolor = outmesh.add_property_map<Mesh::Vertex_index, CGAL::Color>("v:color").first;

//    if(path.extension().string() == ".off"){
////        CGAL::IO::write_OFF(filename,outmesh,CGAL::parameters::vertex_color_map(vcolor).face_color_map(fcolor));
//        CGAL::IO::write_OFF(filename,outmesh,CGAL::parameters::vertex_color_map(vcolor));
//    }
//    else if(path.extension().string() == ".ply"){
//        //        CGAL::IO::write_PLY(filename,outmesh,CGAL::parameters::vertex_color_map(vcolor).face_color_map(fcolor));
//        CGAL::IO::write_PLY(filename,outmesh,CGAL::parameters::vertex_color_map(vcolor));
//    }
//    else{
//        _logger->error("{} is not a valid file ending. Only '.ply' and '.off' files are allowed.", path.extension().string());
//        return 1;
//    }
    return 0;

}

int SMesh::save_mesh(const string filename){

    Mesh mesh;
    return save_mesh(filename,mesh);

}



void SMesh::_color_mesh_by_region(){

    auto fcolor = _mesh.add_property_map<Mesh::Face_index, CGAL::Color>("f:color",CGAL::white()).first;

    for(auto face : _mesh.faces()){
        // get the poly that made this face
//        auto poly = _polygon_to_face.find(face)->first;
        // get the color of this poly
        auto region = _face_to_region[face];
        auto color = _colors[region];
        // set the color to this face
        fcolor[face] = color;
    }
}


#include <random>
void SMesh::_color_mesh_boundary(Mesh& mesh, int seed){


    Mesh::Property_map<Mesh::Vertex_index, CGAL::Color>
        color_map = mesh.add_property_map<Mesh::Vertex_index, CGAL::Color>("v:color",CGAL::white()).first;

    vector<Mesh::Halfedge_index> boundaries;
    PMP::extract_boundary_cycles(mesh,back_inserter(boundaries));

    default_random_engine generator(seed);
    uniform_int_distribution<int> uniform_distribution(100, 225);
    for(auto he0 : boundaries){

        unsigned char r = 0, g = 0, b = 0;
        r = uniform_distribution(generator);
        g = uniform_distribution(generator);
        b = uniform_distribution(generator);
        auto color = CGAL::IO::Color(r,g,b);

        for (auto he : CGAL::halfedges_around_face(he0, mesh)){
            color_map[target(he, mesh)] = color;
        }
    }
}

struct Visitor : public CGAL::Polygon_mesh_processing::Default_orientation_visitor
{
    // from CGAL example here: https://doc.cgal.org/latest/Polygon_mesh_processing/Polygon_mesh_processing_2orient_polygon_soup_example_8cpp-example.html
    vector<Mesh::Vertex_index>& _duplicate_points;
    Visitor(vector<Mesh::Vertex_index>& duplicate_points) : _duplicate_points(duplicate_points){}

    void duplicated_vertex(std::size_t v1, std::size_t v2)
    {
        Mesh::Vertex_index vi1(v1);
        Mesh::Vertex_index vi2(v2);
        _duplicate_points.push_back(vi1);
        _duplicate_points.push_back(vi2);
//        std::cout << "The vertex " << v1 << " has been duplicated, its new id is " << v2 << "." << std::endl;
    }
};
int SMesh::soup_to_mesh_no_repair(){

    _mesh.clear();
    _polygon_to_face.clear();
    boost::associative_property_map<std::map<int, Mesh::Face_index>> ptf(_polygon_to_face);
    boost::associative_property_map<std::map<int, Mesh::Vertex_index>> ptv(_point_to_vertex);

    // These 3 are bundled in repair_polygon_soup
    PMP::merge_duplicate_points_in_polygon_soup(_points, _polygons);
    PMP::merge_duplicate_polygons_in_polygon_soup(_points, _polygons);


//    PMP::repair_polygon_soup(_points, _polygons);
    vector<Mesh::Vertex_index> duplicate_points;
    Visitor visitor(duplicate_points);
    PMP::orient_polygon_soup(_points, _polygons, CGAL::parameters::visitor(visitor));

//    PMP::duplicate_non_manifold_edges_in_polygon_soup(_points,_polygons);
    PMP::polygon_soup_to_polygon_mesh(_points, _polygons, _mesh,
                                      PMP::parameters::polygon_to_face_map(ptf).point_to_vertex_map(ptv));

    _vertex_is_corner = _mesh.add_property_map<Mesh::Vertex_index, bool>("v:corner",false).first;
    _vcolor = _mesh.add_property_map<Mesh::Vertex_index, CGAL::Color>("v:color",CGAL::white()).first;

    for(auto vid : duplicate_points){
//        _vertex_is_corner[vid] = true;
//        _vcolor[vid] = CGAL::blue();
        _vertex_is_corner[_point_to_vertex[vid]] = true;
        _vcolor[_point_to_vertex[vid]] = CGAL::blue();
    }



//    for(auto vert : _mesh.vertices()){
//        if(PMP::is_non_manifold_vertex(vert,_mesh))
//            cout << "found non maifold vertex" << endl;
//    }

    for(auto fid : _mesh.faces()){
        auto pid = _polygon_to_face.find(fid)->first;
        auto region = _polygon_to_region[pid];
        _face_to_region[fid] = region;
        _region_to_faces[region].push_back(fid);
    }

    _get_corner_vertices();

    return 0;

}

int SMesh::soup_to_mesh(const bool triangulate, const bool stitch_borders){

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

    vector<Mesh::Halfedge_index> boundaries;
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

/// the include gives a warning, so I'll comment that part of the code. it is not working anyway.
//#include <CGAL/Polygon_mesh_processing/remesh_planar_patches.h>

//int SMesh::remesh_planar_patches(const int triangulate){

//    if(!CGAL::is_triangle_mesh(_mesh))
//        PMP::triangulate_faces(_mesh);

//    _simplified_mesh.clear();

//    PMP::remesh_planar_patches(_mesh,_simplified_mesh,PMP::parameters::do_not_triangulate_faces(!triangulate));

//    return 0;

//}

//int SMesh::remesh_almost_planar_patches(const int triangulate){

//    _logger->info("Input: ");
//    _logger->info("Vertices: {}",_mesh.number_of_vertices());
//    _logger->info("Edges: {}",_mesh.number_of_edges());
//    _logger->info("Faces: {}",_mesh.number_of_faces());


//    boost::associative_property_map<map<Mesh::Face_index, int>> ftr(_face_to_region);

//    map<Mesh::Vertex_index,int> vertex_to_corner;
//    boost::associative_property_map<map<Mesh::Vertex_index,int>> vtc(vertex_to_corner);


//    int nb_regions = _region_to_polygons.size();

//    map<Mesh::Edge_index,bool> edge_is_constrained;
//    for(Mesh::Edge_index e : _mesh.edges())
//        edge_is_constrained.insert(pair<Mesh::Edge_index,bool>(e, false));
//    boost::associative_property_map<map<Mesh::Edge_index,bool>> eic(edge_is_constrained);

//    //////// //////// //////// //////// //////// //////// //////// //////// //////// //////// ////////
//    //////// nb_corners is always 0, so the whole thing doesn't work
//    int nb_corners = PMP::detect_corners_of_regions(_mesh,ftr,nb_regions,vtc,PMP::parameters::maximum_angle(1).
//                                   maximum_distance(1).edge_is_constrained_map(eic));
//    //////// //////// //////// //////// //////// //////// //////// //////// //////// //////// ////////

//    _simplified_mesh.clear();

//    bool succes = PMP::remesh_almost_planar_patches(_mesh,_simplified_mesh,nb_regions,nb_corners,ftr,vtc,eic,PMP::parameters::do_not_triangulate_faces(!triangulate));

//    if(succes){
//        _logger->info("Simplification succeeded");
//    }
//    else{
//        _logger->info("Simplification did not succeed");
//    }

//    _logger->info("Output: ");
//    _logger->info("Vertices: {}",_simplified_mesh.number_of_vertices());
//    _logger->info("Edges: {}",_simplified_mesh.number_of_edges());
//    _logger->info("Faces: {}",_simplified_mesh.number_of_faces());

//    return 0;
//}

void SMesh::_get_corner_vertices(){

    for(auto vert : _mesh.vertices()){

        unordered_set<int> regions;
        CGAL::Face_around_target_circulator<Mesh> face(_mesh.halfedge(vert),_mesh), done(face);
        do{
            regions.insert(_face_to_region[*face]);
            face++;
        }while(face != done);
        if(regions.size()>2){
            _vertex_is_corner[vert] = true;
            _vcolor[vert] = CGAL::red();
        }
    }
}

void SMesh::_save_region_mesh(const vector<Mesh::Face_index>& region,const string name){

    Mesh region_mesh;

    for(auto fid : region){
        vector<Mesh::vertex_index> polygon;
        for(auto vid : _mesh.vertices_around_face(_mesh.halfedge(fid))){
            auto nvid = region_mesh.add_vertex(_mesh.point(vid));
            polygon.push_back(nvid);
        }
        region_mesh.add_face(polygon);
    }
    string outfile = "/home/rsulzer/data/reconbench/simplify_test/surface"+name+".off";
    save_mesh(outfile,region_mesh);


}


CF::CF(SMesh& smesh) : _smesh(smesh)
{};

template < typename Path, typename Graph >
void CF::cycle(const Path& p, const Graph& g)
{
    // from here: https://stackoverflow.com/a/71630444/20795095

    vector<Mesh::Vertex_index> cycle_full;
    vector<Mesh::Vertex_index> cycle_simplified;
    // Get the property map containing the vertex indices
    // so we can print them.
    typedef typename boost::property_map< Graph, boost::vertex_index_t >::const_type IndexMap;
    IndexMap indices = boost::get(boost::vertex_index, g);

    // Iterate over path printing each vertex that forms the cycle.
    typename Path::const_iterator i, end = p.end();
    for (i = p.begin(); i != end; ++i)
    {
        auto id = boost::get(indices, *i);
        Mesh::Vertex_index vi(id);
        if(_smesh._vertex_is_corner[vi])
            cycle_simplified.push_back(vi);
        cycle_full.push_back(vi);
    }
    cycle_simplified.push_back(cycle_simplified[0]);
    _smesh._cycles_simplified.push_back(cycle_simplified);
    cycle_full.push_back(cycle_full[0]);
    _smesh._cycles_full.push_back(cycle_full);
}

int SMesh::remesh_planar_regions(const bool triangulate, const bool simplify_edges){


    auto cdt = SMesh_CDT(_mesh);

    vector<Mesh> region_meshes;
    vector<vector<Mesh::Vertex_index>> region_corners;
    int rid = 0;
    CF cf(*this);
    for(auto region : _region_to_faces){

        if(_debug_export)
            _save_region_mesh(region.second, to_string(region.first));

        region_corners.clear();

        vector<Mesh::Halfedge_index> borders;
        PMP::border_halfedges(region.second,_mesh,back_inserter(borders));

        Graph graph;
        for(auto border : borders){
            boost::add_edge(_mesh.source(border),_mesh.target(border),graph);
        }

        _cycles_full.clear();
        _cycles_simplified.clear();
        boost::tiernan_all_cycles(graph,cf);
        if(simplify_edges){
            for(int i = 0; i < _cycles_simplified.size(); i+=2){
                // tiernan_all_cycles returns each cycle in both directions, so here I only keep every second cycle/patch
                region_corners.push_back(_cycles_simplified[i]);
            }

        }
        else{
            for(int i = 0; i < _cycles_full.size(); i+=2){
                // tiernan_all_cycles returns each cycle in both directions, so here I only keep every second cycle/patch
                region_corners.push_back(_cycles_full[i]);
            }
        }

        // mesh the region
        Mesh region_mesh;
        if(region_corners.size() == 1 && !triangulate){
            vector<Mesh::vertex_index> polygon;
            for(int i = 0; i < region_corners[0].size() - 1; i++){
                auto nvid = region_mesh.add_vertex(_mesh.point(region_corners[0][i]));
                polygon.push_back(nvid);
            }
            region_mesh.add_face(polygon);
        }
        else if(region_corners.size() > 1 || triangulate){
            region_mesh = cdt._get_boundary_cdt_of_region_mesh(_planes[rid],region_corners);
        }
        else{
            _logger->warn("Empy region boundary");
        }

        region_meshes.push_back(region_mesh);

        if(_debug_export){
            string outfile = "/home/rsulzer/data/reconbench/simplify_test/surface"+to_string(region.first)+"s"+".off";
            save_mesh(outfile,region_mesh);
        }
        rid++;
    }

    _merge_region_meshes(region_meshes);

    return 0;

}


int SMesh::_merge_region_meshes(const vector<Mesh>& meshes){

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
int SMesh::load_soup_from_npz(const string filename){

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









