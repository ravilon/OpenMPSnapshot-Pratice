#ifndef ZFP_DIJKSTRA_HPP
#define ZFP_DIJKSTRA_HPP
#pragma once

#include <vector>

// This function calculates the shortest path from a source node to all other nodes
// in a graph using Dijkstra's algorithm. The function returns a vector of integers
// where each element represents the shortest distance from the source node to the
// corresponding node in the graph. If there is no path from the source node to a
// particular node, the distance is set to -1.
template <typename T>
std::vector<int> dijkstra(const std::vector<std::vector<T>>& graph, int src);

#endif // ZFP_DIJKSTRA_HPP
