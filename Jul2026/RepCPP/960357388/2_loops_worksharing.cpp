// Worksharing - Loops (#pragma omp for or #pragma omp parallel for)

// Concept: The most common use case. Distributes the iterations of a for loop across the team of threads within a parallel region.
// #pragma omp parallel for: A convenient shorthand combining #pragma omp parallel and #pragma omp for.

#include <iostream>
#include <vector>
#include <thread>
#include <omp.h>
#include <chrono> // For timing comparison

int main() {
    std::cout << "--- Parallel For Example ---" << std::endl;
    const int SIZE = 20; // Small size for visible output, increase for performance demo
    std::vector<int> data(SIZE);

    omp_set_num_threads(4); // Request 4 threads

    std::cout << "Initializing vector elements in parallel..." << std::endl;

    // Combine parallel region creation and loop worksharing
    #pragma omp parallel for
    for (int i = 0; i < SIZE; ++i) {
        int thread_id = omp_get_thread_num();
        // The loop iterations are automatically divided among threads.
        // Variable 'i' is implicitly private to each iteration/thread handling it.
        data[i] = thread_id * 100 + i;

        // Use critical section for clean output if needed (demonstration only)
        #pragma omp critical
        std::cout << "Thread " << thread_id << " processed index " << i
                  << ", set data[" << i << "] = " << data[i] << std::endl;

        // Simulate some work per iteration
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } // Implicit barrier here: master thread waits until all iterations are done

    std::cout << "Parallel for finished. Vector data:" << std::endl;
    // for (int i = 0; i < SIZE; ++i) {
    //     std::cout << data[i] << " ";
    // }
    // std::cout << std::endl;

    return 0;
}