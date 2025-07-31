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
#include <omp.h>
#include <functional>

#include "quicksort.hpp"

// Partition function for quicksort
template <typename T>
int partition(std::vector<T>& arr, int left, int right, 
              std::function<bool(T, T)> eval) 
{
    T pivot = arr[right];
    int partitionIndex = left;

    for (int i = left; i < right; ++i) 
    {
        if (eval(arr[i], pivot)) 
        {
            std::swap(arr[i], arr[partitionIndex]);
            ++partitionIndex;
        }
    }
    std::swap(arr[partitionIndex], arr[right]);
    return partitionIndex;
}

#ifdef ZFP_RECURSIVE_FORM
// Recursive form of quicksort with OpenMP and custom evaluation function
template <typename T>
void quickSort(std::vector<T>& arr, int left, int right, 
               std::function<bool(T, T)> eval) 
{
    if (left < right) 
    {
        int partitionIndex = partition(arr, left, right, eval);

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                quickSort(arr, left, partitionIndex - 1, eval);
            }
            #pragma omp section
            {
                quickSort(arr, partitionIndex + 1, right, eval);
            }
        }
    }
}
#else
// Iterative form of quicksort with OpenMP and custom evaluation function
template <typename T>
void quickSort(std::vector<T>& arr, int left, int right, 
               std::function<bool(T, T)> eval) 
{
    if (left >= right) return;

    struct Range {
        int left, right;
    };

    std::vector<Range> stack;
    stack.push_back({left, right});

    while (!stack.empty()) 
    {
        Range range = stack.back();
        stack.pop_back();

        left = range.left;
        right = range.right;

        if (left < right) 
        {
            int partitionIndex = partition(arr, left, right, eval);

            stack.push_back({left, partitionIndex - 1});
            stack.push_back({partitionIndex + 1, right});
        }
    }
}
#endif

// Explicit template instantiation
template void quickSort<int>(std::vector<int>&, int, int, 
                             std::function<bool(int, int)>);
template void quickSort<float>(std::vector<float>&, int, int, 
                               std::function<bool(float, float)>);
template void quickSort<double>(std::vector<double>&, int, int, 
                                std::function<bool(double, double)>);
template void quickSort<long>(std::vector<long>&, int, int,
                              std::function<bool(long, long)>);
