#include "kmb.h"
#include "dijkstra.h"
#include "prim_mst.h"
#include <algorithm>
#include <set>
#include <omp.h>

void KMB::outerParallel()
{
  Graph graph1(vector<vector<Edge>>(graph.adjacency_list.size(), vector<Edge>()), {});

  set<pair<int, int>> added_edges;
  map<pair<int, int>, vector<Edge>> shortest_path_edges;

#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < terminals.size(); ++i)
  {
    int u = terminals[i];
    for (int j = i + 1; j < terminals.size(); ++j)
    {
      int v = terminals[j];

      if (u < 0 || u >= graph.adjacency_list.size())
      {
        cout << "Invalid terminal " << u << endl;
        continue;
      }
      if (u >= v)
      {
        continue;
      }

      vector<int> path;
      vector<Edge> edges;

      Dijkstra dijkstra;
      dijkstra.find_shortest_path(graph, u, v, path, edges);

#pragma omp critical
      {
        shortest_path_edges[{u, v}] = edges;
        if (added_edges.find({u, v}) == added_edges.end() && added_edges.find({v, u}) == added_edges.end())
        {
          graph1.adjacency_list[u].push_back({v, dijkstra.dist[v], u});
          added_edges.insert({u, v});
        }
      }
    }
  }

  PrimMST prim;
  Graph T1 = prim.find_mst_parallel(graph1, terminals[0]);

  set<int> vertices;
  for (auto &edge : shortest_path_edges)
  {
    for (Edge &vertex : edge.second)
    {
      vertices.insert(vertex.to);
    }
  }

  Graph graph2(vector<vector<Edge>>(graph.adjacency_list.size(), vector<Edge>()), {});
  for (auto &edge : shortest_path_edges)
  {
    for (Edge &vertex : edge.second)
    {
      graph2.adjacency_list[vertex.to].push_back({vertex.from, vertex.length, vertex.to});
    }
  }

  Graph steiner = prim.find_mst_parallel(graph2, terminals[0]);
  cout << "VALUE " << steiner.getGraphWeight() << endl;
}
