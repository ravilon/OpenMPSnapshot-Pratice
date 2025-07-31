#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

#define MAX 1000000 // Size of the array

int main() {
    std::vector<double> A(MAX, 1.0); // Initialize array with some values
    double ave = 0.0;
    
    auto start_time = std::chrono::high_resolution_clock::now(); // Start time measurement

    #pragma omp parallel for reduction(+:ave)
    for (int i = 0; i < MAX; i++) {
        ave += A[i];
    }
    
    ave = ave / MAX;
    
    auto end_time = std::chrono::high_resolution_clock::now(); // End time measurement
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    std::cout << "Average: " << ave << std::endl;
    std::cout << "Execution Time: " << elapsed.count() << " seconds" << std::endl;
    return 0;
}