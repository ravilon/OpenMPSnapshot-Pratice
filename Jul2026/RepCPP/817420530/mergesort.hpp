#ifndef ZFP_MERGESORT_HPP
#define ZFP_MERGESORT_HPP
#pragma once

#include <functional>
#include <vector>

// This implementation of the merge sort algorithm uses OpenMP
// to parallelize the sorting process. The algorithm can be
// implemented in a recursive or iterative form.
// Define ZFP_RECURSIVE_FORM to use the recursive form.
template <typename T>
void mergeSort(std::vector<T>& arr, int left, int right, 
               std::function<bool(T, T)> eval);

#endif // ZFP_MERGESORT_HPP
