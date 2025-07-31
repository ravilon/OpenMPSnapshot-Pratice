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

#include <random>
#include <omp.h>

#include "montecarlo_simulation.hpp"

template <typename T>
T monteCarloPi(int num_samples) 
{
    int count = 0;

    #pragma omp parallel reduction(+:count)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(0.0, 1.0);

        #pragma omp for
        for (int i = 0; i < num_samples; ++i) {
            T x = dis(gen);
            T y = dis(gen);
            if (x * x + y * y <= 1.0) {
                count++;
            }
        }
    }

    return static_cast<T>(4.0) * count / num_samples;
}

// Explicit instantiation of the function
template float monteCarloPi<float>(int num_samples);
template double monteCarloPi<double>(int num_samples);
