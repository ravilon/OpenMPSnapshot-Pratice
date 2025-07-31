#include "base.h"
#include <sstream>
#include <algorithm>
#include <climits>
#include <omp.h>

void load_graph(const std::string &filename, std::map<int, int> &nodes, std::map<std::pair<int, int>, int> &distances)
{
    std::ifstream infile(filename);
    if (!infile)
    {
        std::cerr << "Erro ao abrir o arquivo" << std::endl;
        return;
    }

    int num_nodes;
    infile >> num_nodes;

    nodes.clear();
    nodes[0] = 0;

    for (int i = 1; i < num_nodes; ++i)
    { // Start from 1 because the first node is the origin
        int id, pedido;
        infile >> id >> pedido;
        nodes[id] = pedido;
    }

    int num_edges;
    infile >> num_edges;

    distances.clear();
    int node1, node2, distance;
    for (int i = 0; i < num_edges; ++i)
    {
        infile >> node1 >> node2 >> distance;
        distances[{node1, node2}] = distance;
    }

    infile.close();
}

std::vector<std::vector<int>> generatePermutations(const std::map<int, int> &locations)
{
    std::vector<std::vector<int>> permutations;
    std::vector<int> indexes;

    std::transform(locations.begin(), locations.end(), std::back_inserter(indexes), [](const auto &pair) { return pair.first; });

    int n = indexes.size();
    int num_permutations = 1;
    for (int i = 1; i <= n; ++i)
    {
        num_permutations *= i;
    }

    for (int i = 0; i < num_permutations; ++i)
    {
        permutations.push_back(indexes);
        std::next_permutation(indexes.begin(), indexes.end());
    }

    return permutations;
}

std::vector<std::vector<int>> generatePermutationsParallel(const std::map<int, int> &locations)
{
    std::vector<std::vector<int>> permutations;
    std::vector<int> indexes;

    std::transform(locations.begin(), locations.end(), std::back_inserter(indexes), [](const auto &pair) { return pair.first; });

    int n = indexes.size();
    int num_permutations = 1;
    for (int i = 1; i <= n; ++i)
    {
        num_permutations *= i;
    }

    permutations.resize(num_permutations);

#pragma omp parallel for
    for (int i = 0; i < num_permutations; ++i)
    {
        std::vector<int> local_indexes = indexes;
        for (int j = 0; j < i; ++j)
        {
            std::next_permutation(local_indexes.begin(), local_indexes.end());
        }
        permutations[i] = local_indexes;
    }

    return permutations;
}

std::vector<std::vector<int>> generatePermutationsParallelOptimized(const std::map<int, int> &locations)
{
    std::vector<std::vector<int>> permutations;
    std::vector<int> indexes;

    std::transform(locations.begin(), locations.end(), std::back_inserter(indexes), [](const auto &pair) { return pair.first; });

    std::sort(indexes.begin(), indexes.end());

    do
    {
        permutations.push_back(indexes);
    } while (std::next_permutation(indexes.begin(), indexes.end()));

    int num_permutations = permutations.size();
    std::vector<std::vector<int>> parallel_permutations(num_permutations);

#pragma omp parallel for
    for (int i = 0; i < num_permutations; ++i)
    {
        parallel_permutations[i] = permutations[i];
    }

    return parallel_permutations;
}

void printPermutations(const std::vector<std::vector<int>> &permutations)
{
    std::cout << "Permutações:" << std::endl;

    for (const auto &permutation : permutations)
    {
        for (int node : permutation)
        {
            std::cout << node << " ";
        }
        std::cout << std::endl;
    }
}

void printPaths(const std::vector<std::vector<int>> &possiblePaths)
{
    std::cout << "Caminhos possíveis:" << std::endl;
    for (const auto &path : possiblePaths)
    {
        for (int distance : path)
        {
            std::cout << distance << " ";
        }
        std::cout << std::endl;
    }
}

void printPath(const std::vector<int> &path, const std::string &text, int cost)
{
    std::cout << text << ": ";
    for (int node : path)
    {
        std::cout << node << " ";
    }
    std::cout << " with cost: " << cost << std::endl;
}

std::vector<std::vector<int>> generatePossiblePaths(std::vector<std::vector<int>> permutations, const std::map<std::pair<int, int>, int> &distances, const std::map<int, int> &nodes, int maxCapacity)
{
    std::vector<std::vector<int>> possiblePaths;
    int numPermutations = permutations.size();

    for (int i = 0; i < numPermutations; ++i)
    {
        std::vector<int> path;
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

std::vector<std::vector<int>> generatePossiblePathsParallel(const std::vector<std::vector<int>> &permutations, const std::map<std::pair<int, int>, int> &distances, const std::map<int, int> &nodes, int maxCapacity)
{
    std::vector<std::vector<int>> possiblePaths;
    int numPermutations = permutations.size();

    // Redimensiona o vetor de caminhos possíveis para que possa ser acessado em paralelo
    possiblePaths.resize(numPermutations);

#pragma omp parallel for
    for (int i = 0; i < numPermutations; ++i)
    {
        std::vector<int> path;
        int capacity = 0;

        std::vector<int> perm = permutations[i];

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

int findClosestNode(int node, const std::set<int> &unvisitedNodes, const std::map<int, int> &nodes, const std::map<std::pair<int, int>, int> &distances)
{
    int closestNode = -1;
    int minDistance = INT_MAX;

    std::vector<int> unvisitedVector(unvisitedNodes.begin(), unvisitedNodes.end());

#pragma omp parallel for shared(node, unvisitedVector, distances) reduction(min : minDistance)
    for (size_t i = 0; i < unvisitedVector.size(); ++i)
    {
        int candidate = unvisitedVector[i];
        if (candidate != node)
        {
            auto distIt = distances.find({node, candidate});
            if (distIt != distances.end())
            {
#pragma omp critical
                {
                    int distance = distIt->second;
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
