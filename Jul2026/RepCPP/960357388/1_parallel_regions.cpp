// Parallel Regions (#pragma omp parallel)

// Concept: The fundamental directive. The block of code following this pragma is duplicated and executed by a team of threads created by the OpenMP runtime.

// Runtime Functions: You often use functions from <omp.h> within these regions to get information like the thread ID or the total number of threads.

#include <iostream>
#include <omp.h> // OpenMP header

int main() {
    std::cout << "--- Parallel Region Example ---" << std::endl;

    // Set the number of threads explicitly (optional, runtime often defaults based on cores)
    omp_set_num_threads(4); // Request 4 threads for subsequent parallel regions

    #pragma omp parallel // Start a parallel region
    {
        // This block is executed by multiple threads concurrently

        int thread_id = omp_get_thread_num(); // Get unique ID for this thread (0 to N-1)
        int num_threads = omp_get_num_threads(); // Get total number of threads in the team

        // Each thread executes this print statement
        std::cout << "Hello from thread " << thread_id
                  << " out of " << num_threads << " threads." << std::endl;

        // Note: Variables declared inside the parallel region are private to each thread
        int private_var = thread_id * 10;

        #pragma omp barrier // Optional: Wait here until all threads reach this point

        // This print only happens after all threads hit the barrier
        #pragma omp critical // Avoid garbled output from cout (see Synchronization)
        std::cout << "Thread " << thread_id << " passed the barrier with private_var = "
                  << private_var << std::endl;

    } // End of parallel region - threads synchronize and are typically put to sleep or destroyed

    std::cout << "Parallel region finished." << std::endl;
    return 0;
}

// Compile (GCC/Clang): g++ 1_parallel_regions.cpp -o bin/parallel_regions -fopenmp; ./bin/parallel_regions
