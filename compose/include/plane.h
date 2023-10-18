#pragma once

#include <cgal_typedefs.h>

using namespace std;

template <typename Kernel>
class Plane{
public:

    Plane(vector<double> vec);

    vector<double> _vector;
    int _max_coord;
    typename Kernel::Plane_3 _cgal;

    typedef typename Kernel::Point_3 Point;
    typedef typename Kernel::Point_2 Point2;
    typedef typename CGAL::Surface_mesh<Point> Mesh;


    Point2 project(Point);
    void color_mesh_by_max_coord(Mesh& mesh);

};

