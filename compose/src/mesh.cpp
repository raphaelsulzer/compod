//#include <iostream>
//#include <mesh.h>
//#include <cdt.h>

//#include <boost_typedefs.h>

//#include <CGAL/Polygon_mesh_processing/region_growing.h>
//#include <CGAL/Polygon_mesh_processing/orient_polygon_soup_extension.h>

//using namespace std;
//namespace fs = boost::filesystem;

//template <typename Kernel>
//SMesh<Kernel>::SMesh(int verbosity, bool debug_export){

//    _debug_export = debug_export;


//    if(spdlog::get("SMesh")){
//        _logger = spdlog::get("SMesh");
//    }
//    else{
//        _logger = spdlog::stdout_color_mt("SMesh");
//    }

//    if(verbosity == 0)
//        _logger->set_level(spdlog::level::warn);
//    else if(verbosity == 1)
//        _logger->set_level(spdlog::level::info);
//    else if(verbosity == 2)
//        _logger->set_level(spdlog::level::debug);
//    else
//        _logger->set_level(spdlog::level::off);

//    spdlog::set_pattern("[%H:%M:%S] [%n] [%l] %v");

//}


//template <typename Kernel>
//int SMesh<Kernel>::save_mesh(const string filename, Mesh& outmesh){

//    auto path = fs::path(filename);

//    if(!fs::is_directory(path.parent_path()))
//        fs::create_directories(path.parent_path());

//    if(outmesh.number_of_faces()>0){
//        _logger->debug("Save surface mesh to {}",filename);
//    }
//    else if(_mesh.number_of_faces()>0){
//        outmesh = _mesh;
//        _logger->debug("Save surface mesh to {}",filename);
//    }
//    else{
//        _logger->error("No mesh available to save. First run soup_to_mesh().");
//        return 1;
//    }

//    CGAL::IO::write_polygon_mesh(filename,outmesh);
//    return 0;

//}

//template <typename Kernel>
//int SMesh<Kernel>::save_mesh(const string filename){

//    Mesh mesh;
//    return save_mesh(filename,mesh);

//}

//template <typename Kernel>
//int SMesh<Kernel>::_merge_region_meshes(const vector<Mesh>& meshes){

//    vector<Point> new_points;
//    vector<Polygon> new_polys;
//    int n = 0;
//    for(auto mesh : meshes){

//        for(auto p : mesh.points()){
//            new_points.push_back(p);
//        }

//        for(auto fi : mesh.faces()){

//            Polygon poly;
//            CGAL::Vertex_around_face_circulator<Mesh> vcirc(mesh.halfedge(fi), mesh), done(vcirc);
//            do{
//                poly.push_back(*vcirc++ + n);
//            }while (vcirc != done);
//            new_polys.push_back(poly);
//        }
//        n+=mesh.number_of_vertices();

//    }

//    _polygons = new_polys;
//    _points = new_points;


//    return 0;

//}

//#include <xtensor-io/xnpz.hpp>
//#include <xtensor/xnpy.hpp>
//#include <xtensor/xarray.hpp>
//#include <xtensor/xfixed.hpp>
//#include <xtensor/xio.hpp>
//#include <xtensor/xtensor.hpp>
//template <typename Kernel>
//int SMesh<Kernel>::load_soup_from_npz(const string filename){

//    /////// with xtensor
//    auto arr = xt::load_npz(filename);

//    if(arr.find("points") == arr.end()){
//        _logger->error("No points array found in {}", filename);
//        return 1;
//    }
//    if(arr.find("polygons") == arr.end()){
//        _logger->error("No polygons array found in {}", filename);
//        return 1;
//    }


//    auto pts = arr["points"].cast<double>();
//    auto polys = arr["polygons"].cast<int>();
//    auto polygon_regions = arr["polygon_regions"].cast<int>();
//    auto planes = arr["planes"].cast<float>();
//    auto colors = arr["colors"].cast<int>();

//    _points.clear();
//    _polygons.clear();
//    _colors.clear();
//    _polygon_to_region.clear();
//    _region_to_polygons.clear();
//    _planes.clear();


//    for (size_t i = 0; i < pts.shape(0); i++){
//        _points.push_back(Point(pts(i,0),pts(i,1),pts(i,2)));
//    }


//    int n = 0;
//    for (size_t i = 0; i < polys.shape(0); i++){
//        Polygon poly;
//        for(size_t j = 0; j < polys(i); j++){
//            poly.push_back(j+n);
//        }
//        n+=poly.size();
//        _region_to_polygons[polygon_regions[i]].push_back(poly);
//        _polygons.push_back(poly);
//    }


//    for (size_t i = 0; i < planes.shape(0); i++){
//        _planes.push_back(Plane({planes(i,0),planes(i,1),planes(i,2),planes(i,3)}));
//    }

//    for (size_t i = 0; i < colors.shape(0); i++){
//        _colors.push_back(CGAL::Color(colors(i,0),colors(i,1),colors(i,2)));
//    }

//    for (size_t i = 0; i < polygon_regions.shape(0); i++){
//        _polygon_to_region.push_back(polygon_regions(i));
//    }

//    int min = *min_element(_polygon_to_region.begin(), _polygon_to_region.end());
//    int max = *max_element(_polygon_to_region.begin(), _polygon_to_region.end());
//    assert(min >= 0);
//    assert(max < _planes.size());

//    return 0;


//}









