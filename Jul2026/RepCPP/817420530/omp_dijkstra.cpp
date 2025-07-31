// Copyright 2023 Paolo Fabio Zaino
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vector>
#include <limits.h>
#include <omp.h>

#include "dijkstra.hpp"

int minDistance(const std::vector<int>& dist, const std::vector<bool>& sptSet) 
{
    int min = INT_MAX, min_index = -1;

    for (size_t v = 0; v < dist.size(); v++)
        if (!sptSet[v] && dist[v] <= min) {
            min = dist[v];
            min_index = v;
        }

    return min_index;
}

template <typename T>
std::vector<int> dijkstra(const std::vector<std::vector<T>>& graph, int src) 
{
    int n = graph.size();
    if (n == 0) return {};

    std::vector<int> dist(n, INT_MAX);
    std::vector<bool> sptSet(n, false);

    dist[src] = 0;

    for (int count = 0; count < n - 1; count++) 
    {
        int u = minDistance(dist, sptSet);
        if (u == -1) break;

        sptSet[u] = true;

        #pragma omp parallel for
        for (int v = 0; v < n; v++)
            if (!sptSet[v] && graph[u][v] && dist[u] != INT_MAX && dist[u] + graph[u][v] < dist[v])
                dist[v] = dist[u] + graph[u][v];
    }

    return dist;
}

// Explicit template instantiation
template std::vector<int> dijkstra<int>(const std::vector<std::vector<int>>&, int);
template std::vector<int> dijkstra<float>(const std::vector<std::vector<float>>&, int);
template std::vector<int> dijkstra<double>(const std::vector<std::vector<double>>&, int);
template std::vector<int> dijkstra<long>(const std::vector<std::vector<long>>&, int);
template std::vector<int> dijkstra<unsigned int>(const std::vector<std::vector<unsigned int>>&, int);
