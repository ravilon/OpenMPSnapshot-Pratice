#ifndef ZFP_QUICKSORT_HPP
#define ZFP_QUICKSORT_HPP
#pragma once

#include <functional>
#include <vector>

// This implementation of the quicksort algorithm uses OpenMP
// to parallelize the sorting process. The algorithm can be
// implemented in a recursive or iterative form.
// Define ZFP_RECURSIVE_FORM to use the recursive form.
template <typename T>
void quickSort(std::vector<T>& arr, int left, int right, 
               std::function<bool(T, T)> eval);

#endif // ZFP_QUICKSORT_HPP
