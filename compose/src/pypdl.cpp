#include <iostream>
#include <cgal_typedefs.h>
#include <pypdl.h>
#include <random>
#include <sys/stat.h>
#include <CGAL/IO/polygon_soup_io.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>


pyPDL::pyPDL(int n_test_points)
{
    _n_test_points = n_test_points;
}

pyPDL::~pyPDL()
{
    if (_tree != nullptr) {
        delete _tree;
    }
}

int pyPDL::load_mesh(const string filename){

//    _logger = spdlog::get("PyLabeler");

    struct stat buffer;
    if(!stat(filename.c_str(), &buffer) == 0){
        cout << "ERROR: " << filename << " does not exist" << endl;
//        _logger->error("ERROR: {} does not exist",filename);
        return 1;
    }
    CGAL::IO::read_polygon_mesh(filename, _gt_mesh);
    if(_gt_mesh.number_of_faces() == 0){
        cout << "ERROR: " << filename << " has no faces" << endl;
        cout << "Will try to read a soup and make a mesh out of it" << endl;
    }
    else{
        _init_tree();
        return 0;
    }

    _gt_mesh.clear();
    vector<EPICK::Point_3> pts;
    vector<vector<int>> polys;

    CGAL::IO::read_polygon_soup(filename, pts,polys);
    CGAL::Polygon_mesh_processing::polygon_soup_to_polygon_mesh(pts,polys,_gt_mesh);

    if(_gt_mesh.number_of_faces() == 0){
        cout << "ERROR: " << filename << " has no faces. Cannot do anything." << endl;
        return 1;
    }
    else{
        if(!CGAL::is_triangle_mesh(_gt_mesh))
            CGAL::Polygon_mesh_processing::triangulate_faces(_gt_mesh);

        string outfile = filename.substr(0,filename.size()-4)+"_repaired.off";
        cout << "Mesh loading worked. Will export the mesh as " << outfile << endl;
        CGAL::IO::write_polygon_mesh(outfile,_gt_mesh);

        _init_tree();
        return 0;
    }
}


void pyPDL::_init_tree(){

    _tree = new AABB_Tree(faces(_gt_mesh).first, faces(_gt_mesh).second, _gt_mesh);
    _tree->accelerate_distance_queries();
    // Initialize the point-in-polyhedron tester

}


// Export colored sampled points for debug
#include <CGAL/property_map.h>
#include <CGAL/IO/write_ply_points.h>
typedef std::tuple<EPICK::Point_3, CGAL::Color> PC;
typedef CGAL::Nth_of_tuple_property_map<0, PC> Point_map;
typedef CGAL::Nth_of_tuple_property_map<1, PC> Color_map;
void pyPDL::export_test_points(const string filename){

    ofstream f(filename, std::ios::binary);
    CGAL::IO::set_binary_mode(f); // The PLY file will be written in the binary format
    CGAL::IO::write_PLY_with_properties(f, _all_sampled_points,
                                        CGAL::make_ply_point_writer(Point_map()),
                                        std::make_tuple(Color_map(),
                                                        CGAL::IO::PLY_property<unsigned char>("red"),
                                                        CGAL::IO::PLY_property<unsigned char>("green"),
                                                        CGAL::IO::PLY_property<unsigned char>("blue"),
                                                        CGAL::IO::PLY_property<unsigned char>("alpha")));
}

const float pyPDL::_label(vector<EPICK::Point_3>& inexact_polyhedron_points){

    // random sampler
    CGAL::Random random(42);
    Delaunay Dt;
    vector<EPICK::Point_3> sampled_points;

    auto inside_tester = Point_inside(*_tree);

    Dt.insert(inexact_polyhedron_points.begin(),inexact_polyhedron_points.end());

    uniform_int_distribution<int> color(0,255);
    double red = color(_generator),green = color(_generator),blue = color(_generator);
    double pin=0,pout=0;
    int tetin, tetout;
    double tet_vol;
    for(auto aci = Dt.finite_cells_begin(); aci != Dt.finite_cells_end(); aci++){
        tetin=0,tetout=0;
        EPICK::Tetrahedron_3 current_tet = Dt.tetrahedron(aci);
        tet_vol = CGAL::volume(current_tet.vertex(0),current_tet.vertex(1),current_tet.vertex(2),current_tet.vertex(3));
//        tet_vol = color(_generator);
        CGAL::Random_points_in_tetrahedron_3<EPICK::Point_3> tet_point_sampler(current_tet, random);
        sampled_points.clear();
        CGAL::cpp11::copy_n(tet_point_sampler, _n_test_points, std::back_inserter(sampled_points));
        for(auto const& sampled_point : sampled_points){
            if(inside_tester(sampled_point) == CGAL::ON_BOUNDED_SIDE){
                _all_sampled_points.push_back(make_tuple(sampled_point,CGAL::red()));
                tetin+=1;
            }
            else{
                _all_sampled_points.push_back(make_tuple(sampled_point,CGAL::blue()));
                tetout+=1;
            }
        }
        pin=pin+(tetin*tet_vol);
        pout=pout+(tetout*tet_vol);
    }

    return pin/(pout+pin);
}

double
pyPDL::label_one_cell(const nb::ndarray<double, nb::shape<-1, 3>>& points){

    vector<EPICK::Point_3> cgpoints;
    for(int j = 0; j < points.shape(0); j++)
        cgpoints.push_back(EPICK::Point_3(points(j,0),points(j,1),points(j,2)));
    return this->_label(cgpoints);
}


vector<double>
pyPDL::label_cells(const nb::ndarray<int, nb::shape<-1>>& points_len, const nb::ndarray<double, nb::shape<-1, 3>>& points){

    _all_sampled_points.clear();

    int k=0;

    vector<double> occs;
    vector<EPICK::Point_3> cgpoints;

    for(int i = 0; i < points_len.shape(0); i++){
        cgpoints.clear();
        for(int j = 0; j < points_len(i); j++){
            cgpoints.push_back(EPICK::Point_3(points(k,0),points(k,1),points(k,2)));
            k+=1;
        }
        occs.push_back(this->_label(cgpoints));
    }

    return occs;

}

NB_MODULE(libPYPDL, m) {
    nb::class_<pyPDL>(m, "pdl")
            .def(nb::init<int>(),"n_test_points"_a = 100)
            .def("load_mesh", &pyPDL::load_mesh, "filename"_a, "Load a polygon soup.")
            .def("label_cells", &pyPDL::label_cells, "number_of_cell_points"_a, "cell_vertices"_a, "Generate a polygon mesh from the polygon soup.")
            .def("label_one_cell", &pyPDL::label_one_cell, "cell_vertices"_a, "Generate a polygon mesh from the polygon soup.")
            .def("export_test_points", &pyPDL::export_test_points, "filename"_a, "Export the test points colored by occupancy.");
}

