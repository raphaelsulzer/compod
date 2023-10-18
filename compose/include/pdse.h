//#pragma once


//using namespace std;

//#include "spdlog/spdlog.h"
//#include "spdlog/sinks/stdout_color_sinks.h"
//#include "mesh.h"

//template <typename Kernel>
//class PDSE{
//public:


//    PDSE(const int verbosity = 1, const bool debug_export = false);
//    ~PDSE() { spdlog::drop("PDSE");}

////    template <typename Kernel>
////    int make_mesh(const string filename, string outfilename = "",
////                  const bool triangulate = false, const bool stitch_borders = true);

////    template <typename Kernel>
////    int make_simplified_mesh(const string filename, string outfilename = "",
////                  const bool triangulate = false,
////                  const bool simplify_edges = true);


//    int make_mesh(const string filename, string outfilename = "",
//                  const bool triangulate = false, const bool stitch_borders = true)
//    {

//        if(outfilename.empty())
//            outfilename = filename.substr(filename.size() - 3) + "off";

//        auto smesh = SMesh<Kernel>(_verbosity,_debug_export);
//        smesh.load_soup_from_npz(filename);
//        smesh.soup_to_mesh(triangulate, stitch_borders);
//    //    smesh.soup_to_mesh2(triangulate, stitch_borders);

//        smesh.save_mesh(outfilename);

//        return 0;

//    }

//    int make_simplified_mesh(const string filename, string outfilename = "",
//                  const bool triangulate = false,
//                  const bool simplify_edges = true)
//    {


//        if(outfilename.empty())
//            outfilename = filename.substr(filename.size() - 3) + "off";

//        auto smesh = SMesh<Kernel>(_verbosity,_debug_export);
//        smesh.load_soup_from_npz(filename);


//        smesh.soup_to_mesh_no_repair();

//        if(_debug_export){
//            string debugfilename = "/home/rsulzer/data/reconbench/simplify_test/surface_colored.off";
//            smesh.save_mesh(debugfilename);
//        }


//        smesh.remesh_planar_regions(triangulate, simplify_edges);

//        smesh.soup_to_mesh(triangulate);

//        smesh.save_mesh(outfilename);

//        return 0;

//    }


//private:
//    bool _debug_export;
//    int _verbosity;
//    shared_ptr<spdlog::logger> _logger;
//};

