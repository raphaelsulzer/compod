#pragma once

using namespace std;

#include <CGAL/Surface_mesh.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>


//typedef CGAL::Exact_predicates_exact_constructions_kernel Kernel;
//typedef Kernel::FT Rational;

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;

typedef Kernel::Point_3 Point;
typedef Kernel::Point_2 Point2;
typedef vector<int> Polygon;
typedef CGAL::Surface_mesh<Point> Mesh;
