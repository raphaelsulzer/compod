//#include <iostream>
//#include <random>
//#include <plane.h>
//#include <cdt.h>
//#include <boost/filesystem.hpp>
//#include <CGAL/Constrained_Delaunay_triangulation_2.h>
//#include <CGAL/Triangulation_face_base_with_info_2.h>

//template <typename Kernel>
//SMesh_CDT<Kernel>::SMesh_CDT(Mesh& mesh){
//    _mesh = mesh;
//}

//// Marking inside and outside domains of a polygon, from CGAL example: https://doc.cgal.org/latest/Triangulation_2/index.html#title30

//template <typename Kernel>
//void SMesh_CDT<Kernel>::mark_domains(CDT& ct,
//             typename CDT::Face_handle start,
//             int index,
//             std::list<typename CDT::Edge>& border )
//{
//  if(start->info().nesting_level != -1){
//    return;
//  }
//  std::list<typename CDT::Face_handle> queue;
//  queue.push_back(start);
//  while(! queue.empty()){
//    typename CDT::Face_handle fh = queue.front();
//    queue.pop_front();
//    if(fh->info().nesting_level == -1){
//      fh->info().nesting_level = index;
//      for(int i = 0; i < 3; i++){
//        typename CDT::Edge e(fh,i);
//        typename CDT::Face_handle n = fh->neighbor(i);
//        if(n->info().nesting_level == -1){
//          if(ct.is_constrained(e)) border.push_back(e);
//          else queue.push_back(n);
//        }
//      }
//    }
//  }
//}
////explore set of facets connected with non constrained edges,
////and attribute to each such set a nesting level.
////We start from facets incident to the infinite vertex, with a nesting
////level of 0. Then we recursively consider the non-explored facets incident
////to constrained edges bounding the former set and increase the nesting level by 1.
////Facets in the domain are those with an odd nesting level.
//template <typename Kernel>
//void SMesh_CDT<Kernel>::mark_domains(CDT& cdt)
//{
//  for(typename CDT::Face_handle f : cdt.all_face_handles()){
//    f->info().nesting_level = -1;
//  }
//  std::list<typename CDT::Edge> border;
//  mark_domains(cdt, cdt.infinite_face(), 0, border);
//  while(! border.empty()){
//    typename CDT::Edge e = border.front();
//    border.pop_front();
//    typename CDT::Face_handle n = e.first->neighbor(e.second);
//    if(n->info().nesting_level == -1){
//      mark_domains(cdt, n, e.first->info().nesting_level+1, border);
//    }
//  }
//}

//template <typename Kernel>
//typename SMesh_CDT<Kernel>::Mesh SMesh_CDT<Kernel>::_get_boundary_cdt_of_region_mesh(Plane<Kernel>& plane, vector<vector<typename Mesh::vertex_index>>& region_corners){

//    // make a 2D constrained delaunay triangulation of the region
//    CDT cdt;
//    map<typename CDT::Vertex_handle,typename Mesh::Vertex_index> two_d_to_three_d;
//    for(auto patch : region_corners){

//        // insert points
//        for(int i = 0; i < patch.size()-1; i++){
//            Point2 pts = plane.project(_mesh.point(patch[i]));
//            Point2 ptt = plane.project(_mesh.point(patch[i+1]));
//            auto dts = cdt.push_back(pts);
//            two_d_to_three_d[dts] = patch[i];
//            cdt.insert_constraint(pts,ptt);
//        }
//    }
//    mark_domains(cdt);
//    assert(cdt.is_valid());

//    // get back the 3D mesh with a subset of the original vertices
//    Mesh region_mesh;
//    typename CDT::Finite_faces_iterator fit;
//    for(fit = cdt.finite_faces_begin(); fit != cdt.finite_faces_end(); fit++){
//        if( !fit->info().in_domain() )
//            continue;
//        vector<typename Mesh::vertex_index> polygon;
//        typename Mesh::Vertex_index vid;
//        auto face = *fit;
//        for(size_t i = 0; i < 3; i++){
//            // check if there are vertices created by the CDT, can sometimes happen
//            vid = two_d_to_three_d[face.vertex(i)];
//            auto vi = region_mesh.add_vertex(_mesh.point(vid));
//            polygon.push_back(vi);
//        }
//        region_mesh.add_face(polygon);
//    }

//    return region_mesh;
//}
