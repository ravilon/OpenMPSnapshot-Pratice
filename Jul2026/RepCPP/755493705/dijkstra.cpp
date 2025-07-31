#include "dijkstra.h"
#include <queue>
#include <limits.h>
#include <algorithm>

#include <omp.h>

Dijkstra::Dijkstra() {}

using namespace std;

void Dijkstra::find_shortest_path(Graph &graph, int start, int end, vector<int> &path, vector<Edge> &edges)
{
  int n = graph.adjacency_list.size();
  dist.assign(n, INT_MAX);
  prev.assign(n, -1);
  priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
  pq.push({0, start});
  dist[start] = 0;

  while (!pq.empty())
  {
    int u = pq.top().second;
    int d = pq.top().first;
    pq.pop();

    if (d > dist[u])
      continue;

    for (Edge &e : graph.adjacency_list[u])
    {
      int v = e.to;
      int w = e.length;
      if (dist[u] + w < dist[v])
      {
        dist[v] = dist[u] + w;
        prev[v] = u;
        pq.push({dist[v], v});
      }
    }
  }

  // Reconstruct the shortest path
  int current_vertex = end;
  while (current_vertex != -1)
  {
    path.push_back(current_vertex);
    if (prev[current_vertex] != -1)
    {
      edges.push_back(Edge(prev[current_vertex], dist[current_vertex] - dist[prev[current_vertex]], current_vertex));
    }
    current_vertex = prev[current_vertex];
  }
  reverse(path.begin(), path.end());   // Reverse to get the correct order
  reverse(edges.begin(), edges.end()); // Reverse to get the correct order
}

void Dijkstra::find_shortest_path_parallel(Graph &graph, int start, int end, vector<int> &path, vector<Edge> &edges)
{
  int n = graph.adjacency_list.size();
  dist.assign(n, INT_MAX);
  prev.assign(n, -1);
  priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
  pq.push({0, start});
  dist[start] = 0;

  while (!pq.empty())
  {
    int u, d;
#pragma omp critical
    {
      u = pq.top().second;
      d = pq.top().first;
      pq.pop();
    }

    if (d > dist[u])
      continue;

#pragma omp parallel for
    for (size_t i = 0; i < graph.adjacency_list[u].size(); ++i)
    {
      Edge &e = graph.adjacency_list[u][i];
      int v = e.to;
      int w = e.length;

      int new_dist;
#pragma omp critical
      {
        new_dist = dist[u] + w;
      }

      if (new_dist < dist[v])
      {
#pragma omp critical
        {
          dist[v] = new_dist;
          prev[v] = u;
          pq.push({dist[v], v});
        }
      }
    }
  }

  // Reconstruct the shortest path
  int current_vertex = end;
  while (current_vertex != -1)
  {
    path.push_back(current_vertex);
    if (prev[current_vertex] != -1)
    {
      edges.push_back(Edge(prev[current_vertex], dist[current_vertex] - dist[prev[current_vertex]], current_vertex));
    }
    current_vertex = prev[current_vertex];
  }
  reverse(path.begin(), path.end());   // Reverse to get the correct order
  reverse(edges.begin(), edges.end()); // Reverse to get the correct order
}
