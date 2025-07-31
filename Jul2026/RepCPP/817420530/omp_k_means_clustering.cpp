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
#include <cmath>
#include <limits>
#include <omp.h>
#include <random>

#include "k_means_clustering.hpp"

template <typename T>
std::vector<int> kmeans(const std::vector<std::vector<T>>& data, int k, 
                        int max_iterations) 
{
    int n = data.size();
    if (n == 0) return {};  // Handle the case of empty data
    int d = data[0].size();

    std::vector<std::vector<T>> centroids(k, std::vector<T>(d, 0));
    std::vector<int> labels(n, 0);
    std::vector<int> counts(k, 0);

    // Initialize centroids using k-means++ (or any other initialization strategy)
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, n - 1);
    for (int i = 0; i < k; ++i) {
        centroids[i] = data[dist(rng)];
    }

    for (int iter = 0; iter < max_iterations; ++iter) {
        // Assign labels
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            T min_dist = std::numeric_limits<T>::max();
            int min_index = 0;
            for (int j = 0; j < k; ++j) {
                T dist = 0;
                for (int l = 0; l < d; ++l) {
                    dist += (data[i][l] - centroids[j][l]) * (data[i][l] - centroids[j][l]);
                }
                if (dist < min_dist) {
                    min_dist = dist;
                    min_index = j;
                }
            }
            labels[i] = min_index;
        }

        // Recompute centroids
        std::vector<std::vector<T>> new_centroids(k, std::vector<T>(d, 0));
        std::vector<int> new_counts(k, 0);

        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            int label = labels[i];
            #pragma omp atomic
            new_counts[label]++;
            for (int j = 0; j < d; ++j) {
                #pragma omp atomic
                new_centroids[label][j] += data[i][j];
            }
        }

        #pragma omp parallel for
        for (int j = 0; j < k; ++j) {
            if (new_counts[j] > 0) {
                for (int l = 0; l < d; ++l) {
                    new_centroids[j][l] /= new_counts[j];
                }
            } else {
                new_centroids[j] = data[dist(rng)];  // Reinitialize empty centroid
            }
        }

        centroids.swap(new_centroids);
    }

    return labels;
}

// Explicit template instantiation
template std::vector<int> kmeans<int>(const std::vector<std::vector<int>>&, int, int);
template std::vector<int> kmeans<float>(const std::vector<std::vector<float>>&, int, int);
template std::vector<int> kmeans<double>(const std::vector<std::vector<double>>&, int, int);
template std::vector<int> kmeans<long>(const std::vector<std::vector<long>>&, int, int);
template std::vector<int> kmeans<unsigned int>(const std::vector<std::vector<unsigned int>>&, int, int);
