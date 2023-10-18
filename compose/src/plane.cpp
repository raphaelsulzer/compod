#include <plane.h>

template <typename Kernel>
Plane<Kernel>::Plane(vector<double> vec){

    _vector = vec;
    auto abs_vec = {abs(vec[0]),abs(vec[1]),abs(vec[2])};

    auto max = max_element(abs_vec.begin(),abs_vec.end());
    _max_coord = distance(abs_vec.begin(), max); // absolute index of max

    _cgal = Kernel::Plane_3(vec[0],vec[1],vec[2],vec[3]);
}

template <typename Kernel>
typename Plane<Kernel>::Point2 Plane<Kernel>::project(Point pt3){

    if(_max_coord == 0){
        return Point2(pt3.y(),pt3.z());
    }
    else if(_max_coord == 1){
        return Point2(pt3.x(),pt3.z());
    }
    else if(_max_coord == 2){
        return Point2(pt3.x(),pt3.y());
    }
    else{
        cerr << "Invalid _max_coord" << endl;
        return Point2(-999999,-999999);
    }
}

//template <typename Kernel>
//void Plane<Kernel>::color_mesh_by_max_coord(Mesh& mesh){

//    if(_max_coord == 0){
//        mesh.add_property_map<typename Mesh<Kernel>::Vertex_index, CGAL::Color>("v:color",CGAL::red());
//    }
//    else if(_max_coord == 1){
//        mesh.add_property_map<typename Mesh::Vertex_index, typename CGAL::Color>("v:color",CGAL::green());
//    }
//    else if(_max_coord == 2){
//        mesh.add_property_map<typename Mesh::Vertex_index, typename CGAL::Color>("v:color",CGAL::blue());
//    }
//    else{
//        cerr << "Invalid _max_coord" << endl;
//        return;
//    }
//}
