// Synchronization (#pragma omp critical, #pragma omp atomic)

// Concept: Preventing race conditions when multiple threads need to update shared resources.
// #pragma omp critical [(name)]: Defines a critical section. Only one thread at a time can enter any critical section with the same name (or unnamed if no name is given). Protects complex updates.
// #pragma omp atomic [update | read | write | capture]: Protects a single, simple memory update operation (like x++, x += val, x = val, read/write). Often much faster (lower overhead) than critical for supported operations.

#include <iostream>
#include <vector>
#include <omp.h>

int main() {
    std::cout << "--- Synchronization Example ---" << std::endl;
    long long shared_counter_critical = 0;
    long long shared_counter_atomic = 0;
    const int ITERATIONS = 100000;
    const int NUM_THREADS = 4;

    omp_set_num_threads(NUM_THREADS);

    #pragma omp parallel // Start parallel region
    {
        // --- Critical Section Example ---
        for (int i = 0; i < ITERATIONS; ++i) {
            #pragma omp critical // Only one thread enters this block at a time
            {
                shared_counter_critical++;
                // Could do more complex, non-atomic operations here safely
            }
        }

        // --- Atomic Operation Example ---
        for (int i = 0; i < ITERATIONS; ++i) {
            // Protects only the single increment operation. Often faster.
            #pragma omp atomic update // C++ style: update clause is often optional
            shared_counter_atomic++;
        }
    } // End parallel region

    long long expected_count = (long long)NUM_THREADS * ITERATIONS;
    std::cout << "Expected count:        " << expected_count << std::endl;
    std::cout << "Result (critical):     " << shared_counter_critical << std::endl;
    std::cout << "Result (atomic):       " << shared_counter_atomic << std::endl;
    std::cout << "Critical match:    " << (shared_counter_critical == expected_count ? "Yes" : "No") << std::endl;
     std::cout << "Atomic match:      " << (shared_counter_atomic == expected_count ? "Yes" : "No") << std::endl;


    return 0;
}