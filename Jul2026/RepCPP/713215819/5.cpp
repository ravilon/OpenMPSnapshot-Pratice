#include <omp.h>
#include <stdio.h>
#include <chrono>

int main() {
    const int k = 10000; // Number of iterations

    // Sequential Run
    auto start_time_seq = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < k; ++i) {
        printf("Name: Dhivyesh R K, Register Number: 2021BCS0084\n");
    }

    auto end_time_seq = std::chrono::high_resolution_clock::now();
    auto duration_seq = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_seq - start_time_seq);

    // Parallel Run
    auto start_time_parallel = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (int i = 0; i < k; ++i) {
        printf("Name: Dhivyesh R K, Register Number: 2021BCS0084\n");
    }

    auto end_time_parallel = std::chrono::high_resolution_clock::now();
    auto duration_parallel = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_parallel - start_time_parallel);
    printf("Number of iterations: %d\n",k);
    printf("Sequential Run Time: %ld ms\n", long(duration_seq.count()));
    printf("Parallel Run Time: %ld ms\n", long(duration_parallel.count()));

    return 0;
}

