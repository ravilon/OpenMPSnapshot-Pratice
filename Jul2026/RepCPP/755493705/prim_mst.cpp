#include "prim_mst.h"
#include <set>
#include <queue>

#include <omp.h>

using namespace std;

PrimMST::PrimMST() {}

Graph PrimMST::find_mst(Graph &graph, int start_vertex)
{

  int n = graph.adjacency_list.size();
  adjacency_list.assign(n, vector<Edge>());

  vector<bool> visited(n, false);
  visited[start_vertex] = true;

  priority_queue<Edge> pq;
  for (Edge &e : graph.adjacency_list[start_vertex])
  {
    pq.push(e);
  }

  vector<Edge> mst_edges;

  while (!pq.empty())
  {

    Edge min_edge = pq.top();
    pq.pop();
    int u = min_edge.to;
    if (!visited[u])
    {
      visited[u] = true;
      mst_edges.push_back(min_edge);

      for (Edge &e : graph.adjacency_list[u])
      {
        if (!visited[e.to])
        {
          pq.push(e);
        }
      }
    }
  }

  map<pair<int, int>, int> mst_weights;
  for (Edge &e : mst_edges)
  {
    mst_weights[{e.to, e.length}] = e.length;
    adjacency_list[e.from].push_back({e.to, e.length, e.from});
  }

  return Graph(adjacency_list, mst_weights);
}

Graph PrimMST::find_mst_parallel(Graph &graph, int start_vertex)
{
  int n = graph.adjacency_list.size();
  adjacency_list.assign(n, vector<Edge>());

  vector<bool> visited(n, false);
  visited[start_vertex] = true;

  priority_queue<Edge> pq;
  for (Edge &e : graph.adjacency_list[start_vertex])
  {
    pq.push(e);
  }

  vector<Edge> mst_edges;

  // Use a flag to control loop termination
  bool continue_loop = true;

#pragma omp parallel
  {
    while (continue_loop)
    {
      Edge min_edge;
      bool has_edge = false;

// Only one thread should access the priority queue at a time
#pragma omp critical
      {
        if (!pq.empty())
        {
          min_edge = pq.top();
          pq.pop();
          has_edge = true;
        }
        else
        {
          // If the priority queue is empty, set the flag to exit the loop
          continue_loop = false;
        }
      }

      if (has_edge)
      {
        int u = min_edge.to;
        if (!visited[u])
        {
// Avoid accessing shared data concurrently by multiple threads
#pragma omp critical
          {
            if (!visited[u])
            {
              visited[u] = true;
              mst_edges.push_back(min_edge);
              for (Edge &e : graph.adjacency_list[u])
              {
                if (!visited[e.to])
                {
                  pq.push(e);
                }
              }
            }
          }
        }
      }
    }
  }

  map<pair<int, int>, int> mst_weights;
  for (Edge &e : mst_edges)
  {
    mst_weights[{e.to, e.length}] = e.length;
    adjacency_list[e.from].push_back({e.to, e.length, e.from});
  }

  return Graph(adjacency_list, mst_weights);
}