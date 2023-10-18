#pragma once

using namespace std;

#include <boost/filesystem.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/directed_graph.hpp>
#include <boost/graph/tiernan_all_cycles.hpp>

#include <cgal_typedefs.h>

using Graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS>;
//namespace boost { void renumber_vertex_indices(Graph const&) {} }
namespace boost { void renumber_vertex_indices(Graph const&); }
//typedef boost::graph_traits<Graph>::vertex_descriptor BoostVertex;
//typedef map<Mesh::Vertex_index, BoostVertex> VertexMap;
