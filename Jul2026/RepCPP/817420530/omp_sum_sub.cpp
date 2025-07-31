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

#include "sum_sub.hpp"

// parallelSum function sums all the elements of a vector in parallel
template <typename T>
T parallelSum(const std::vector<T>& vec) {
    T sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < vec.size(); ++i) {
        sum += vec[i];
    }

    return sum;
}

// prefixSum function calculates the prefix sum of a vector in parallel
template <typename T>
void prefixSum(std::vector<T>& arr) {
    int n = arr.size();
    std::vector<T> prefix_sum(n);

    #pragma omp parallel
    {
        T sum = 0;

        #pragma omp for
        for (int i = 0; i < n; ++i) {
            sum += arr[i];
            prefix_sum[i] = sum;
        }
    }

    arr = prefix_sum;
}

// parallelSub function subtracts all the elements of a vector in parallel
template <typename T>
T parallelSub(const std::vector<T>& vec) {
    T sub = 0;

    #pragma omp parallel for reduction(+:sub)
    for (size_t i = 0; i < vec.size(); ++i) {
        sub += vec[i];
    }

    return -sub;
}

// prefixSub function calculates the prefix subtraction of a vector in parallel
template <typename T>
void prefixSub(std::vector<T>& arr) {
    int n = arr.size();
    std::vector<T> prefix_sub(n);

    #pragma omp parallel
    {
        T sub = 0;

        #pragma omp for
        for (int i = 0; i < n; ++i) {
            sub -= arr[i];
            prefix_sub[i] = sub;
        }
    }

    arr = prefix_sub;
}

// Explicit template instantiation
template void prefixSum<int>(std::vector<int>&);
template void prefixSum<float>(std::vector<float>&);
template void prefixSum<double>(std::vector<double>&);
template void prefixSum<long>(std::vector<long>&);

template int parallelSum<int>(const std::vector<int>&);
template float parallelSum<float>(const std::vector<float>&);
template double parallelSum<double>(const std::vector<double>&);
template long parallelSum<long>(const std::vector<long>&);

template void prefixSub<int>(std::vector<int>&);
template void prefixSub<float>(std::vector<float>&);
template void prefixSub<double>(std::vector<double>&);
template void prefixSub<long>(std::vector<long>&);

template int parallelSub<int>(const std::vector<int>&);
template float parallelSub<float>(const std::vector<float>&);
template double parallelSub<double>(const std::vector<double>&);
template long parallelSub<long>(const std::vector<long>&);
