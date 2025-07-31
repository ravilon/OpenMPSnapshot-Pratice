// Copyright 2019 zhaofeng-shu33
#pragma once
/// \file
/// \brief triangle counting algorithm
///
#include <vector>

#include "lemontc/config.h"

namespace lemon {
    //! use edge iteration method to count the triangle
    //! \return the exact number of triangle in the graph
    int64_t triangle_count(const Graph& G, int total_edge);
    //! compute the degree for each node and save them to degree_list
    int collect_degree_info(const Graph& G,
            std::vector<int>* degree_list, int node_size);
    //! use node iteration method to count the triangle
    //! \return the exact number of triangle in the graph
    int64_t triangle_count_vertex_iteration(const Graph& G,
                const std::vector<int>& degree_list, int max_degree);
}  // namespace lemon
