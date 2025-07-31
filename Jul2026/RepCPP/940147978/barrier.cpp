
#include <iostream>
#include <omp.h>
#include <vector>

// Simulated expensive calculations
double big_calc1(int id) {
    return id * 2.5; // Example computation
}

double big_calc2(int id, const std::vector<double>& A) {
    return A[id] * 1.1; // Example computation using A[]
}

int main() {
    const int NUM_THREADS = 4;
    std::vector<double> A(NUM_THREADS, 0.0);
    std::vector<double> B(NUM_THREADS, 0.0);

    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int id = omp_get_thread_num();

        // Step 1: Compute A[id]
        A[id] = big_calc1(id);

        // Ensure all threads finish Step 1 before moving on
        #pragma omp barrier  

        // Step 2: Compute B[id] using A[]
        B[id] = big_calc2(id, A);
    }

    // Output results
    std::cout << "Results:\n";
    for (int i = 0; i < NUM_THREADS; i++) {
        std::cout << "B[" << i << "] = " << B[i] << std::endl;
    }

    return 0;
}
