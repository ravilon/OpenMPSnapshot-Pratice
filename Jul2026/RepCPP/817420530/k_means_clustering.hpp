#ifndef ZFP_K_MEANS_CLUSTERING_HPP
#define ZFP_K_MEANS_CLUSTERING_HPP
#pragma once

#include <vector>
#include <cmath>

// This function performs the k-means clustering algorithm on a dataset. The function
// returns a vector of integers where each element represents the cluster to which the
// corresponding data point belongs. The function takes the following parameters:
// - data: a vector of vectors representing the data points to be clustered
// - k: the number of clusters to create
// - max_iterations: the maximum number of iterations to run the algorithm (default = 100)
template <typename T>
std::vector<int> kmeans(const std::vector<std::vector<T>>& data, int k, 
                        int max_iterations = 100);

#endif // ZFP_K_MEANS_CLUSTERING_HPP
