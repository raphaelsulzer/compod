#pragma once

#include <cgal_typedefs.h>


using namespace std;


#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#include <plane.h>
#include <mesh.h>



struct FaceInfo2
{
  FaceInfo2(){}
  int nesting_level = 0;
  bool in_domain(){
    return nesting_level%2 == 1;
  }
};

//typedef CGAL::Exact_predicates_inexact_constructions_kernel             K;
typedef CGAL::Triangulation_vertex_base_2<Kernel>                            Vb;
typedef CGAL::Triangulation_face_base_with_info_2<FaceInfo2,Kernel>          Fbb;
typedef CGAL::Constrained_triangulation_face_base_2<Kernel,Fbb>              Fb;
typedef CGAL::Triangulation_data_structure_2<Vb,Fb>                     TDS;
//typedef CGAL::No_constraint_intersection_tag                            Itag;
typedef CGAL::No_constraint_intersection_requiring_constructions_tag    Itag;
//typedef CGAL::Exact_predicates_tag                                      Itag;
typedef CGAL::Constrained_Delaunay_triangulation_2<Kernel, TDS, Itag>    CDT;


class SMesh_CDT{
public:

    SMesh_CDT(Mesh& mesh);

    Mesh _mesh;

    struct FaceInfo;
    void mark_domains(CDT& ct,
                 CDT::Face_handle start,
                 int index,
                 std::list<CDT::Edge>& border );
    void mark_domains(CDT& cdt);

    Mesh _get_boundary_cdt_of_region_mesh(Plane& plane, vector<vector<Mesh::vertex_index>>& region_corners);

};



