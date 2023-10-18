#include <iostream>
#include <pdse.h>
#include <mesh.h>

using namespace std;

namespace fs = boost::filesystem;

PDSE::PDSE(const int verbosity, const bool debug_export){

    _debug_export = debug_export;
    _verbosity = verbosity;


    if(spdlog::get("PDSE")){
        _logger = spdlog::get("PDSE");
    }
    else{
        _logger = spdlog::stdout_color_mt("PDSE");
    }


    if(verbosity == 0)
        _logger->set_level(spdlog::level::warn);
    else if(verbosity == 1)
        _logger->set_level(spdlog::level::info);
    else if(verbosity == 2)
        _logger->set_level(spdlog::level::debug);
    else
        _logger->set_level(spdlog::level::off);

    spdlog::set_pattern("[%H:%M:%S] [%n] [%l] %v");

}



template <typename Kernel>
int PDSE::make_mesh(const string filename, string outfilename,
                    const bool triangulate, const bool stitch_borders){

    if(outfilename.empty())
        outfilename = filename.substr(filename.size() - 3) + "off";

    auto smesh = SMesh<Kernel>(_verbosity,_debug_export);
    smesh.load_soup_from_npz(filename);
    smesh.soup_to_mesh(triangulate, stitch_borders);
//    smesh.soup_to_mesh2(triangulate, stitch_borders);

    smesh.save_mesh(outfilename);

    return 0;

}

template <typename Kernel>
int PDSE::make_simplified_mesh(const string filename, string outfilename,
                    const bool triangulate,
                    const bool simplify_edges){


    if(outfilename.empty())
        outfilename = filename.substr(filename.size() - 3) + "off";

    auto smesh = SMesh<Kernel>(_verbosity,_debug_export);
    smesh.load_soup_from_npz(filename);


    smesh.soup_to_mesh_no_repair();

    if(_debug_export){
        string debugfilename = "/home/rsulzer/data/reconbench/simplify_test/surface_colored.off";
        smesh.save_mesh(debugfilename);
    }


    smesh.remesh_planar_regions(triangulate, simplify_edges);

    smesh.soup_to_mesh(triangulate);

    smesh.save_mesh(outfilename);

    return 0;

}







