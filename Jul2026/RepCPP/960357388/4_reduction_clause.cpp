// Reduction Clause (reduction(operator:variable))

// Concept: Safely combines values from different threads into a single result using a specified operator (e.g., +, *, max, min, &, |). 
// OpenMP handles creating private copies, doing the partial calculation, and combining them correctly at the end.
// Use Case: Calculating sums, products, finding min/max in parallel loops.

#include <iostream>
#include <vector>
#include <numeric> // For std::accumulate (serial sum)
#include <omp.h>

int main() {
    std::cout << "--- Reduction Example ---" << std::endl;
    const int SIZE = 10000;
    std::vector<int> data(SIZE);
    for(int i=0; i<SIZE; ++i) data[i] = i + 1; // Fill with 1, 2, ..., SIZE

    long long parallel_sum = 0; // Variable to store the final reduced sum
    omp_set_num_threads(4);

    // Calculate sum in parallel using reduction
    // Each thread gets a private 'parallel_sum' initialized to 0.
    // At the end, all private sums are added (+) to the original 'parallel_sum'.
    #pragma omp parallel for reduction(+:parallel_sum)
    for (int i = 0; i < SIZE; ++i) {
        parallel_sum += data[i]; // Add to thread's private copy
    }
    // Implicit barrier and reduction happens here

    // Calculate serial sum for comparison
    long long serial_sum = std::accumulate(data.begin(), data.end(), 0LL);

    std::cout << "Parallel sum: " << parallel_sum << std::endl;
    std::cout << "Serial sum:   " << serial_sum << std::endl;
    std::cout << "Sums match:   " << (parallel_sum == serial_sum ? "Yes" : "No") << std::endl;

    return 0;
}