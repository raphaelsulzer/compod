//#include <iostream>
//#include <pdse.h>
//#include <cgal_typedefs.h>
//#include <boost/filesystem.hpp>
//#include <CGAL/Polygon_mesh_processing/border.h>
//#include <CGAL/Constrained_Delaunay_triangulation_2.h>
//#include <CGAL/Triangulation_face_base_with_info_2.h>




//int main(int argc, char *argv[]){

//    string filename;
//    if(argc > 1)
//        filename = argv[1];
//    else
//        filename = "/home/rsulzer/data/reconbench/abspy/2/product-earlystop/anchor/1/surface.npz";

//    string outfilename;
//    bool triangulate;
//    bool stitch_borders;


//    PDSE<EPECK> pdse(2);

//    outfilename = "/home/rsulzer/data/reconbench/simplify_test/surface.off";
//    triangulate = true;
//    stitch_borders = false;
//    pdse.make_mesh(filename,outfilename, triangulate, stitch_borders);

//    outfilename = "/home/rsulzer/data/reconbench/simplify_test/surface_stitched.off";
//    triangulate = true;
//    stitch_borders = true;
//    pdse.make_mesh(filename,outfilename, triangulate, stitch_borders);

//    outfilename = "/home/rsulzer/data/reconbench/simplify_test/surface_simplified.off";
//    triangulate = true;
//    pdse.make_simplified_mesh(filename,outfilename, triangulate);

//    outfilename = "/home/rsulzer/data/reconbench/simplify_test/surface_simplified_noedge.off";
//    triangulate = true;
//    pdse.make_simplified_mesh(filename,outfilename, triangulate, false);

//}
