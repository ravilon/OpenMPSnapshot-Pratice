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

#include "mergesort.hpp"

template <typename T>
void merge(std::vector<T>& arr, int left, int mid, int right, 
           std::function<bool(T, T)> eval) 
{
    int n1 = mid - left + 1;
    int n2 = right - mid;

    std::vector<T> L(n1);
    std::vector<T> R(n2);

    for (int i = 0; i < n1; ++i) 
    {
        L[i] = arr[left + i];
    }
    for (int j = 0; j < n2; ++j) 
    {
        R[j] = arr[mid + 1 + j];
    }

    int i = 0, j = 0, k = left;

    while (i < n1 && j < n2) 
    {
        if (eval(L[i], R[j])) 
        {
            arr[k] = L[i];
            ++i;
        } 
        else 
        {
            arr[k] = R[j];
            ++j;
        }
        ++k;
    }

    while (i < n1) 
    {
        arr[k] = L[i];
        ++i;
        ++k;
    }

    while (j < n2) 
    {
        arr[k] = R[j];
        ++j;
        ++k;
    }
}

// Explicit template instantiation
template void merge<int>(std::vector<int>&, int, int, int, std::function<bool(int, int)>);
template void merge<float>(std::vector<float>&, int, int, int, std::function<bool(float, float)>);
template void merge<double>(std::vector<double>&, int, int, int, std::function<bool(double, double)>);
template void merge<long>(std::vector<long>&, int, int, int, std::function<bool(long, long)>);

#ifdef ZFP_RECURSIVE_FORM
template <typename T>
void mergeSort(std::vector<T>& arr, int left, int right, 
               std::function<bool(T, T)> eval) 
{
    if (left < right) 
    {
        int mid = left + (right - left) / 2;

        #pragma omp parallel sections
        {
            #pragma omp section
            {
                mergeSort(arr, left, mid, eval);
            }
            #pragma omp section
            {
                mergeSort(arr, mid + 1, right, eval);
            }
        }

        merge(arr, left, mid, right, eval);
    }
}
#else
template <typename T>
void mergeSort(std::vector<T>& arr, int left, int right, 
               std::function<bool(T, T)> eval) 
{
    if (left >= right) return;

    struct Range {
        int left, right;
    };

    std::vector<Range> stack;
    stack.push_back({left, right});

    while (!stack.empty()) {
        Range range = stack.back();
        stack.pop_back();

        left = range.left;
        right = range.right;

        if (left < right) {
            int mid = left + (right - left) / 2;

            stack.push_back({left, mid});
            stack.push_back({mid + 1, right});

            #pragma omp task
            mergeSort(arr, left, mid, eval);
            #pragma omp task
            mergeSort(arr, mid + 1, right, eval);
            #pragma omp taskwait

            merge(arr, left, mid, right, eval);
        }
    }
}
#endif

// Explicit template instantiation
template void mergeSort<int>(std::vector<int>&, int, int, std::function<bool(int, int)>);
template void mergeSort<float>(std::vector<float>&, int, int, std::function<bool(float, float)>);
template void mergeSort<double>(std::vector<double>&, int, int, std::function<bool(double, double)>);
template void mergeSort<long>(std::vector<long>&, int, int, std::function<bool(long, long)>);
