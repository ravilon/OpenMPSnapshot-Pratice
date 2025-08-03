#ifndef ZFP_SUM_SUB_HPP
#define ZFP_SUM_SUB_HPP
#pragma once

#include <vector>

// prefixSum function calculates the prefix sum of a vector in parallel
template <typename T>
void prefixSum(std::vector<T>& vec);

// parallelSum computes the sum of all elements in the vector
template <typename T>
T parallelSum(const std::vector<T>& vec);

// prefixSub function calculates the prefix subtraction of a vector in parallel
template <typename T>
void prefixSub(std::vector<T>& vec);

// parallelSub computes the subtraction of all elements in the vector
template <typename T>
T parallelSub(const std::vector<T>& vec);

#endif // ZFP_SUM_SUB_HPP
