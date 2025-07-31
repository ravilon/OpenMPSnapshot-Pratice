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

vector<int> findBestPath(vector<vector<int>> possiblePaths, const map<pair<int, int>, int> &distances, int &cost)
{
    vector<int> bestPath;
    int minCost = INT_MAX;

    for (int i = 0; i < possiblePaths.size(); i++)
    {
        int pathCost = 0;
        for (int j = 0; j < possiblePaths[i].size() - 1; j++)
        {
            int from = possiblePaths[i][j];
            int to = possiblePaths[i][j + 1];
            int cost_ = distances.at({from, to});
            pathCost += cost_;
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

vector<int> findBestPathParallel(vector<vector<int>> possiblePaths, const map<pair<int, int>, int> &distances, int &cost)
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
            int cost_ = distances.at({from, to});
            pathCost += cost_;
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

vector<int> nearestNeighborSearch(const map<pair<int, int>, int> &distances, const map<int, int> &nodes, int &cost, int maxCapacity)
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

vector<int> nearestNeighborSearchParallel(const map<pair<int, int>, int> &distances, const map<int, int> &nodes, int &cost, int maxCapacity)
{
    vector<int> path;
    cost = 0;
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
                int capacity = 0;
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

int main(int argc, const char *argv[])
{
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <graph_file>" << endl;
        return 1;
    }

    string filePath = argv[1];
    int maxCapacity = 10;
    map<int, int> nodes;
    map<pair<int, int>, int> distances;

    // Load the graph from the provided file path
    load_graph(filePath, nodes, distances);

    // cout << "--- Times ---" << endl;

    auto start = high_resolution_clock::now();
    vector<vector<int>> permutations = generatePermutations(nodes);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "Permutations (S) " << duration.count() << " milli." << endl;

    start = high_resolution_clock::now();
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    cout << "Permutations (P) " << duration.count() << " milli." << endl;

    start = high_resolution_clock::now();
    vector<vector<int>> possiblePaths = generatePossiblePaths(permutations, distances, nodes, maxCapacity);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    cout << "Paths (S) " << duration.count() << " milli." << endl;

    start = high_resolution_clock::now();
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    cout << "Paths (P) " << duration.count() << " milli." << endl;

    int costBestPath = 0;
    start = high_resolution_clock::now();
    vector<int> bestPath = findBestPath(possiblePaths, distances, costBestPath);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    cout << "Best Path (S) " << duration.count() << " milli." << endl;

    int costBestPathParallel = 0;
    start = high_resolution_clock::now();
    vector<int> bestPathParalles = findBestPathParallel(possiblePaths, distances, costBestPathParallel);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    cout << "Best Path (P)  " << duration.count() << " milli." << endl;

    int costNearestNeighbor = 0;
    start = high_resolution_clock::now();
    vector<int> nearestNeighborPath = nearestNeighborSearch(distances, nodes, costNearestNeighbor, maxCapacity);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    cout << "Nearest Neighbor (S) " << duration.count() << " milli." << endl;

    int costNearestNeighborParallel = 0;
    start = high_resolution_clock::now();
    vector<int> nearestNeighborPathParallel = nearestNeighborSearchParallel(distances, nodes, costNearestNeighborParallel, maxCapacity);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    cout << "Nearest Neighbor (P) " << duration.count() << " milli." << endl;

    printPath(bestPath, "Global (S)", costBestPath);
    printPath(bestPathParalles, "Global (P)", costBestPathParallel);
    printPath(nearestNeighborPath, "Nearest Neighbor (S)", costNearestNeighbor);
    printPath(nearestNeighborPathParallel, "Nearest Neighbor (P)", costNearestNeighborParallel);

    return 0;
}
