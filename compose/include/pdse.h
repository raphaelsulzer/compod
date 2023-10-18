#pragma once


using namespace std;

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "mesh.h"

class PDSE{
public:


    PDSE(const int verbosity = 1, const bool debug_export = false);
    ~PDSE() { spdlog::drop("PDSE");}

    template <typename Kernel>
    int make_mesh(const string filename, string outfilename = "",
                  const bool triangulate = false, const bool stitch_borders = true);

    template <typename Kernel>
    int make_simplified_mesh(const string filename, string outfilename = "",
                  const bool triangulate = false,
                  const bool simplify_edges = true);

private:
    bool _debug_export;
    int _verbosity;
    shared_ptr<spdlog::logger> _logger;
};

