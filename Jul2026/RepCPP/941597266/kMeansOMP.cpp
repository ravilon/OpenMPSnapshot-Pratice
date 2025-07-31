/**
 * @file kMeansOMP.cpp
 * @brief OpenMP implementation of the K-means clustering algorithm.
 * @details This file contains the implementation of the `KMeansOMP` class, which utilizes OpenMP for parallel processing 
 *          to perform the K-means clustering algorithm using multiple threads. The class includes methods for initializing 
 *          centroids, updating cluster assignments, recalculating centroids, and synchronizing results across threads. 
 *          This implementation leverages multi-threading to enhance performance in shared-memory environments.
 */


#include "kMeansOMP.hpp"
#include <algorithm>
#include <limits>
#include <chrono>
#include <iostream>

km::KMeansOMP::KMeansOMP(const int& k, const std::vector<Point>& points)
    : KMeansBase(k, points) {}

auto km::KMeansOMP::run() -> void
{
    bool change = true;
    int iter = 0;
    const int iter_max = 1000;

    while (change && iter < iter_max)
    {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        std::vector<int> counts(k, 0);
        change = false;

        counts = std::vector<int>(k, 0);
        #pragma omp parallel
        {

            #pragma omp for
            for (auto& p : points)
            {
                double minDist = std::numeric_limits<double>::max();
                int nearest = 0;
                for (int j = 0; j < k; ++j)
                {
                    double tempDist = p.distance(centroids[j]);
                    if (tempDist < minDist)
                    {
                        minDist = tempDist;
                        nearest = j;
                    }
                }
                if (p.clusterId != nearest)
                {
                    p.clusterId = nearest;
                    change = true;
                }
            }
        }
            std::vector<std::vector<double>> partial_sum(k, {0.0, 0.0, 0.0});
            std::vector<int> cluster_size(k, 0);
            
        #pragma omp parallel
        {
            #pragma omp for
            for (int i = 0; i < k; ++i)
            {
                for (auto& point : points)
                {
                    if (point.clusterId == i)
                    {
                        for (int x = 0; x < 3; x++)
                        {
                            partial_sum[i][x] += point.getFeature_int(x);
                        }
                        cluster_size[i]++;
                    }
                }
            }

            #pragma omp for
            for (int i = 0; i < k; ++i)
            {
                for (int j = 0; j < 3; j++)
                {
                    int result = static_cast<int>(partial_sum[i][j] / cluster_size[i]);
                    centroids[i].setFeature(j, result);
                }
            }
        }
        iter++;
        #pragma omp master
        {
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            std::cout << "Iteration " << iter << " completed in " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << '\n';
            number_of_iterations = iter;
        }
    }
}