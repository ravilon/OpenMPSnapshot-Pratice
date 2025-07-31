#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

#define N 1000000 // Size of the array

int main() {
    std::vector<int> a(N, 1), b(N, 2); // Initialize arrays with some values

    auto start_time = std::chrono::high_resolution_clock::now(); // Start time measurement

    #pragma omp parallel for schedule(dynamic) 
    for (int i = 0; i < N; i += 4) {
        a[i] = a[i] + b[i];
        if (i + 1 < N) a[i + 1] = a[i + 1] + b[i + 1];
        if (i + 2 < N) a[i + 2] = a[i + 2] + b[i + 2];
        if (i + 3 < N) a[i + 3] = a[i + 3] + b[i + 3];
    }
    
    auto end_time = std::chrono::high_resolution_clock::now(); // End time measurement
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    std::cout << "Execution Time: " << elapsed.count() << " seconds" << std::endl;
    return 0;
}
