#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <set>
#include <omp.h>
#include "base.h"
#include <mpi.h>
#include <chrono>
#include <climits>

using namespace std;
using namespace std::chrono;

vector<int> findBestPath(const vector<vector<int>> &possiblePaths, const map<pair<int, int>, int> &distances, int &cost)
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

vector<int> findBestPathParallel(const vector<vector<int>> &possiblePaths, const map<pair<int, int>, int> &distances, int &cost)
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

vector<int> findBestPathParallelMPI(const vector<vector<int>> &possiblePaths, const map<pair<int, int>, int> &distances, int &cost, int rank, int size)
{
    vector<int> bestPath;
    int minCost = INT_MAX;

    int chunkSize = possiblePaths.size() / size;
    int start = rank * chunkSize;
    int end = (rank == size - 1) ? possiblePaths.size() : start + chunkSize;

    // Para evitar usar uma seção crítica, vamos usar variáveis privadas para cada thread
    int localMinCost = INT_MAX;
    vector<int> localBestPath;

#pragma omp parallel
    {
        int threadMinCost = INT_MAX;
        vector<int> threadBestPath;

#pragma omp for nowait
        for (int i = start; i < end; i++)
        {
            int pathCost = 0;
            for (int j = 0; j < possiblePaths[i].size() - 1; j++)
            {
                int from = possiblePaths[i][j];
                int to = possiblePaths[i][j + 1];
                int cost_ = distances.at({from, to});
                pathCost += cost_;
            }

            if (pathCost < threadMinCost)
            {
                threadMinCost = pathCost;
                threadBestPath = possiblePaths[i];
            }
        }

#pragma omp critical
        {
            if (threadMinCost < localMinCost)
            {
                localMinCost = threadMinCost;
                localBestPath = threadBestPath;
            }
        }
    }

    // Usar MPI para reduzir os resultados dos diferentes processos
    MPI_Allreduce(&localMinCost, &minCost, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if (localMinCost == minCost)
    {
        bestPath = localBestPath;
    }

    // Broadcast do melhor caminho encontrado para todos os processos
    int pathSize = bestPath.size();
    MPI_Bcast(&pathSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    bestPath.resize(pathSize);
    MPI_Bcast(bestPath.data(), pathSize, MPI_INT, 0, MPI_COMM_WORLD);

    cost = minCost;
    return bestPath;
}

vector<int> nearestNeighborSearch(const map<pair<int, int>, int> &distances, const map<int, int> &nodes, int &cost, int maxCapacity)
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

    // Executar enquanto ainda houver nós não visitados
    for (size_t i = 0; !unvisitedNodes.empty(); ++i)
    {
        int capacity = 0;
        int nearestNode = findClosestNode(current, unvisitedNodes, nodes, distances);

        if (nearestNode != -1 && capacity + nodes.at(nearestNode) <= maxCapacity)
        {
            path.push_back(nearestNode);
            cost += distances.at({current, nearestNode});
            current = nearestNode;
            unvisitedNodes.erase(nearestNode);
        }
        else
        {
            path.push_back(0);
            cost += distances.at({current, 0});
            current = 0;
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

vector<int> nearestNeighborSearchParallelMPI(const map<pair<int, int>, int> &distances, const map<int, int> &nodes, int &cost, int maxCapacity, int rank, int size)
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

    vector<int> globalPath(path.size());
    MPI_Allreduce(path.data(), globalPath.data(), path.size(), MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    return globalPath;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int maxCapacity = 10;
    map<int, int> nodes;
    map<pair<int, int>, int> distances;
    load_graph("../grafo.txt", nodes, distances);

    // cout << "--- Times ---" << endl;

    auto start = high_resolution_clock::now();
    vector<vector<int>> permutations = generatePermutations(nodes);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    if (rank == 0) cout << "Permutations (S) " << duration.count() << " milli." << endl;

    // start = high_resolution_clock::now();
    // vector<vector<int>> permutationsParallel = generatePermutationsParallelOptimized(nodes);
    // stop = high_resolution_clock::now();
    // duration = duration_cast<milliseconds>(stop - start);
    // if (rank == 0) cout << "Permutations (P) " << duration.count() << " milli." << endl;

    // start = high_resolution_clock::now();
    // vector<vector<int>> possiblePaths = generatePossiblePaths(permutations, distances, nodes, maxCapacity);
    // stop = high_resolution_clock::now();
    // duration = duration_cast<milliseconds>(stop - start);
    // if (rank == 0) cout << "Paths (S) " << duration.count() << " milli." << endl;

    start = high_resolution_clock::now();
    vector<vector<int>> possiblePathsParallel = generatePossiblePathsParallel(permutations, distances, nodes, maxCapacity);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    if (rank == 0) cout << "Paths (P) " << duration.count() << " milli." << endl;

    int costBestPath = 0;
    start = high_resolution_clock::now();
    vector<int> bestPath = findBestPath(possiblePathsParallel, distances, costBestPath);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    if (rank == 0) cout << "Best Path (S) " << duration.count() << " milli." << endl;

    int costBestPathParallel = 0;
    start = high_resolution_clock::now();
    vector<int> bestPathParallel = findBestPathParallel(possiblePathsParallel, distances, costBestPathParallel);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    if (rank == 0) cout << "Best Path (P)  " << duration.count() << " milli." << endl;

    int costBestPathParallelMPI = 0;
    start = high_resolution_clock::now();
    vector<int> bestPathParallelMPI = findBestPathParallelMPI(possiblePathsParallel, distances, costBestPathParallelMPI, rank, size);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    if (rank == 0) cout << "Best Path (MPI)  " << duration.count() << " milli." << endl;

    int costNearestNeighbor = 0;
    start = high_resolution_clock::now();
    vector<int> nearestNeighborPath = nearestNeighborSearch(distances, nodes, costNearestNeighbor, maxCapacity);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    if (rank == 0) cout << "Nearest Neighbor (S) " << duration.count() << " milli." << endl;

    int costNearestNeighborParallel = 0;
    start = high_resolution_clock::now();
    vector<int> nearestNeighborPathParallel = nearestNeighborSearchParallel(distances, nodes, costNearestNeighborParallel, maxCapacity);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    if (rank == 0) cout << "Nearest Neighbor (P) " << duration.count() << " milli." << endl;

    int costNearestNeighborParallelMPI = 0;
    start = high_resolution_clock::now();
    vector<int> nearestNeighborPathParallelMPI = nearestNeighborSearchParallelMPI(distances, nodes, costNearestNeighborParallelMPI, maxCapacity, rank, size);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);

    if (rank == 0)
    {
        cout << "Nearest Neighbor (MPI) " << duration.count() << " milli." << endl;
        printPath(bestPath, "Global (S)", costBestPath);
        printPath(bestPathParallel, "Global (P)", costBestPathParallel);
        printPath(bestPathParallelMPI, "Global (MPI)", costBestPathParallelMPI);
        printPath(nearestNeighborPath, "Nearest Neighbor (S)", costNearestNeighbor);
        printPath(nearestNeighborPathParallel, "Nearest Neighbor (P)", costNearestNeighborParallel);
        printPath(nearestNeighborPathParallelMPI, "Nearest Neighbor (MPI)", costNearestNeighborParallelMPI);
    }

    MPI_Finalize();
    return 0;
}
