#pragma once

#include <cgal_typedefs.h>
#include <type_traits>


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


template <typename Kernel>
class SMesh_CDT{
public:

    typedef typename Kernel::Point_3 Point;
    typedef typename Kernel::Point_2 Point2;
    typedef typename CGAL::Surface_mesh<Point> Mesh;

    typedef CGAL::Triangulation_vertex_base_2<Kernel>                       Vb;
    typedef CGAL::Triangulation_face_base_with_info_2<FaceInfo2,Kernel>     Fbb;
    typedef CGAL::Constrained_triangulation_face_base_2<Kernel,Fbb>         Fb;
    typedef CGAL::Triangulation_data_structure_2<Vb,Fb>                     TDS;


    //typedef CGAL::No_constraint_intersection_tag                            Itag;
    typedef CGAL::No_constraint_intersection_requiring_constructions_tag    Itag;
    typedef CGAL::Exact_predicates_tag                                      Etag;
    using Tag = typename std::conditional<std::is_same<Kernel, EPECK>::value, Etag, Itag>::type;
    typedef CGAL::Constrained_Delaunay_triangulation_2<Kernel, TDS, Tag>    CDT;


    SMesh_CDT(Mesh& mesh);

    Mesh _mesh;

    struct FaceInfo;
    void mark_domains(CDT& ct,
                 typename CDT::Face_handle start,
                 int index,
                 std::list<typename CDT::Edge>& border );
    void mark_domains(CDT& cdt);

    Mesh _get_boundary_cdt_of_region_mesh(Plane<Kernel>& plane, vector<vector<typename Mesh::vertex_index>>& region_corners);

};



