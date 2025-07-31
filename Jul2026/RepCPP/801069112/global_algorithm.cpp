#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <set>
#include <climits>
#include <omp.h>
#include "base.h"

using namespace std;
using namespace std::chrono;

vector<vector<int>> generatePossiblePaths(vector<vector<int>> permutations, map<pair<int, int>, int> distances, map<int, int> &nodes, int maxCapacity)
{
    vector<vector<int>> possiblePaths;
    int numPermutations = permutations.size();
    bool isPathPossible = true;

    for (int i = 0; i < numPermutations; ++i)
    {
        vector<int> path;
        int capacity = 0;

        if (permutations[i][0] != 0)
        {
            permutations[i].insert(permutations[i].begin(), 0);
        }

        int permutationSize = permutations[i].size();

        for (int j = 0; j < permutationSize - 1; ++j)
        {
            int from = permutations[i][j];
            int to = permutations[i][j + 1];
            int nextNodeCapacity = nodes.at(to);

            auto it = distances.find({from, to});

            if (it != distances.end() && capacity + nextNodeCapacity <= maxCapacity)
            {
                path.push_back(from);
                capacity += nextNodeCapacity;
            }
            else
            {
                path.push_back(from);
                if (from != 0)
                {
                    path.push_back(0);
                }
                capacity = nextNodeCapacity;
            }
        }

        path.push_back(permutations[i].back());

        if (path.back() != 0)
        {
            path.push_back(0);
        }

        possiblePaths.push_back(path);
    }

    return possiblePaths;
}

vector<vector<int>> generatePossiblePathsParallel(const vector<vector<int>> &permutations, const map<pair<int, int>, int> &distances, const map<int, int> &nodes, int maxCapacity)
{
    vector<vector<int>> possiblePaths;
    int numPermutations = permutations.size();

    // Redimensiona o vetor de caminhos possíveis para que possa ser acessado em paralelo
    possiblePaths.resize(numPermutations);

// Parallelize the loop with OpenMP
#pragma omp parallel for
    for (int i = 0; i < numPermutations; ++i)
    {
        vector<int> path;
        int capacity = 0;

        vector<int> perm = permutations[i];

        if (perm[0] != 0)
        {
            perm.insert(perm.begin(), 0);
        }

        int permutationSize = perm.size();

        for (int j = 0; j < permutationSize - 1; ++j)
        {
            int from = perm[j];
            int to = perm[j + 1];
            int nextNodeCapacity = nodes.at(to);

            auto it = distances.find({from, to});

            if (it != distances.end() && capacity + nextNodeCapacity <= maxCapacity)
            {
                path.push_back(from);
                capacity += nextNodeCapacity;
            }
            else
            {
                path.push_back(from);
                if (from != 0)
                {
                    path.push_back(0);
                }
                capacity = nextNodeCapacity;
            }
        }

        path.push_back(perm.back());

        if (path.back() != 0)
        {
            path.push_back(0);
        }

        possiblePaths[i] = path;
    }

    return possiblePaths;
}

vector<int> findBestPath(vector<vector<int>> possiblePaths, map<pair<int, int>, int> &distances, int &cost)
{
    vector<int> bestPath;
    int minCost = INT_MAX;
    int numPossiblePaths = possiblePaths.size();

    for (int i = 0; i < possiblePaths.size(); i++)
    {
        int pathCost = 0;
        for (int j = 0; j < possiblePaths[i].size() - 1; j++)
        {
            int from = possiblePaths[i][j];
            int to = possiblePaths[i][j + 1];
            int cost = distances.at({from, to});
            pathCost += cost;
        }
        if (pathCost < minCost)
        {
            minCost = pathCost;
            bestPath = possiblePaths[i];
        }
    }
    cost = minCost;
    return bestPath;
}

vector<int> findBestPathParallel(vector<vector<int>> possiblePaths, map<pair<int, int>, int> &distances, int &cost)
{
    vector<int> bestPath;
    int minCost = INT_MAX;

#pragma omp parallel for
    for (int i = 0; i < possiblePaths.size(); i++)
    {
        int pathCost = 0;
        for (int j = 0; j < possiblePaths[i].size() - 1; j++)
        {
            int from = possiblePaths[i][j];
            int to = possiblePaths[i][j + 1];
            int cost = distances.at({from, to});
            pathCost += cost;
        }
        if (pathCost < minCost)
        {
            minCost = pathCost;
            bestPath = possiblePaths[i];
        }
    }
    cost = minCost;
    return bestPath;
}

int findClosestNode(int node, const set<int> &unvisitedNodes, const map<int, int> &nodes, const map<pair<int, int>, int> &distances)
{
    int closestNode = -1;
    int minDistance = INT_MAX;

    vector<int> unvisitedVector(unvisitedNodes.begin(), unvisitedNodes.end());

#pragma omp parallel for shared(node, unvisitedVector, distances) reduction(min : minDistance)
    for (size_t i = 0; i < unvisitedVector.size(); ++i)
    {
        int candidate = unvisitedVector[i];
        if (candidate != node)
        {
            auto distIt = distances.find({node, candidate});
            if (distIt != distances.end())
            {
                int distance = distIt->second;
#pragma omp critical
                {
                    if (distance < minDistance)
                    {
                        minDistance = distance;
                        closestNode = candidate;
                    }
                }
            }
        }
    }

    return closestNode;
}

vector<int> nearestNeighborSearch(map<pair<int, int>, int> &distances, map<int, int> &nodes, int &cost, int maxCapacity)
{
    vector<int> path;
    cost = 0;
    int capacity = 0;
    int current = 0;
    path.push_back(current);

    set<int> unvisitedNodes;
    for (const auto &node : nodes)
    {
        if (node.first != 0)
        {
            unvisitedNodes.insert(node.first);
        }
    }

    // Executar enquanto ainda houver nós não visitados
    for (size_t i = 0; !unvisitedNodes.empty(); ++i)
    {
        int nearestNode = findClosestNode(current, unvisitedNodes, nodes, distances);

        if (nearestNode != -1 && capacity + nodes.at(nearestNode) <= maxCapacity)
        {
            path.push_back(nearestNode);
            cost += distances.at({current, nearestNode});
            capacity += nodes.at(nearestNode);
            current = nearestNode;
            unvisitedNodes.erase(nearestNode);
        }
        else
        {
            path.push_back(0);
            cost += distances.at({current, 0});
            current = 0;
            capacity = 0;
        }
    }

    if (current != 0)
    {
        path.push_back(0);
        cost += distances.at({current, 0});
    }

    return path;
}

vector<int> nearestNeighborSearchParallel(map<pair<int, int>, int> &distances, map<int, int> &nodes, int &cost, int maxCapacity)
{
    vector<int> path;
    cost = 0;
    int capacity = 0;
    int current = 0;
    path.push_back(current);

    set<int> unvisitedNodes;
    for (const auto &node : nodes)
    {
        if (node.first != 0)
        {
            unvisitedNodes.insert(node.first);
        }
    }

#pragma omp parallel
    {
#pragma omp single
        {
            for (size_t i = 0; !unvisitedNodes.empty(); ++i)
            {
                int nearestNode = -1;

#pragma omp task shared(nearestNode)
                {
                    nearestNode = findClosestNode(current, unvisitedNodes, nodes, distances);
                }

#pragma omp taskwait
                if (nearestNode != -1 && capacity + nodes.at(nearestNode) <= maxCapacity)
                {
#pragma omp critical
                    {
                        path.push_back(nearestNode);
                        cost += distances.at({current, nearestNode});
                        capacity += nodes.at(nearestNode);
                        current = nearestNode;
                        unvisitedNodes.erase(nearestNode);
                    }
                }
                else
                {
#pragma omp critical
                    {
                        path.push_back(0);
                        cost += distances.at({current, 0});
                        current = 0;
                        capacity = 0;
                    }
                }
            }

            if (current != 0)
            {
#pragma omp critical
                {
                    path.push_back(0);
                    cost += distances.at({current, 0});
                }
            }
        }
    }

    return path;
}

void printPermutations(vector<vector<int>> permutations)
{
    cout << "Permutações:" << endl;

    for (const auto &permutation : permutations)
    {
        for (int node : permutation)
        {
            cout << node << " ";
        }
        cout << endl;
    }
}

void printPaths(vector<vector<int>> possiblePaths)
{
    cout << "Caminhos possíveis:" << endl;
    for (const auto &path : possiblePaths)
    {
        for (int distance : path)
        {
            cout << distance << " ";
        }
        cout << endl;
    }
}

void printPath(vector<int> path, string text = "", int cost = 0)
{
    cout << text << endl;
    for (int node : path)
    {
        cout << node << " ";
    }
    cout << " with cost: " << cost << endl;
}

int main()
{
    int maxCapacity = 10;
    map<int, int> nodes;
    map<pair<int, int>, int> distances;
    load_graph("../grafo.txt", nodes, distances);

    cout << "Graph loaded" << endl;

    auto start = high_resolution_clock::now();
    vector<vector<int>> permutations = generatePermutations(nodes);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "Permutations generated in " << duration.count() << " milliseconds" << endl;

    start = high_resolution_clock::now();
    vector<vector<int>> permutationsParallel = generatePermutationsParallelOptimized(nodes);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    cout << "Parallel Permutations generated in " << duration.count() << " milliseconds" << endl;

    start = high_resolution_clock::now();
    vector<vector<int>> possiblePaths = generatePossiblePaths(permutations, distances, nodes, maxCapacity);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    cout << "Possible Paths in " << duration.count() << " milliseconds" << endl;

    start = high_resolution_clock::now();
    vector<vector<int>> possiblePathsParallel = generatePossiblePathsParallel(permutations, distances, nodes, maxCapacity);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    cout << "Parallel Possible Paths generated in " << duration.count() << " milliseconds" << endl;

    int costBestPath = 0;
    start = high_resolution_clock::now();
    vector<int> bestPath = findBestPath(possiblePaths, distances, costBestPath);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    cout << "Best Path in " << duration.count() << " milliseconds" << endl;

    int costBestPathParallel = 0;
    start = high_resolution_clock::now();
    vector<int> bestPathParalles = findBestPathParallel(possiblePaths, distances, costBestPathParallel);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    cout << "Best Path Parallel in  " << duration.count() << " milliseconds" << endl;

    int costNearestNeighbor = 0;
    start = high_resolution_clock::now();
    vector<int> nearestNeighborPath = nearestNeighborSearch(distances, nodes, costNearestNeighbor, maxCapacity);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    cout << "Nearest Neighbor Path in " << duration.count() << " milliseconds" << endl;

    int costNearestNeighborParallel = 0;
    start = high_resolution_clock::now();
    vector<int> nearestNeighborPathParallel = nearestNeighborSearchParallel(distances, nodes, costNearestNeighborParallel, maxCapacity);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    cout << "Nearest Neighbor Path Parallel in " << duration.count() << " milliseconds" << endl;

    printPath(bestPath, "Best Path", costBestPath);
    printPath(nearestNeighborPath, "Nearest Neighbor Path", costNearestNeighbor);
    printPath(nearestNeighborPathParallel, "Nearest Neighbor Path Parallel", costNearestNeighborParallel);

    return 0;
}