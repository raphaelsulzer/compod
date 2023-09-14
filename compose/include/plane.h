#pragma once

#include <cgal_typedefs.h>

using namespace std;





class Plane{
public:

    Plane(vector<double> vec);

    vector<double> _vector;
    int _max_coord;
    Kernel::Plane_3 _cgal;

    Point2 project(Point);
    void color_mesh_by_max_coord(Mesh& mesh);

};

