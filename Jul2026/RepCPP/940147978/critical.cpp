#include <iostream>
#include <omp.h>

// Simulated "big job" function
float big_job(int i) {
    return i * 1.5f; // Example computation
}

// Simulated "consume" function
float consume(float B) {
    return B * 0.8f; // Example processing
}

int main() {
    const int niters = 10;  // Number of iterations
    float res = 0.0f;       // Shared result variable

    #pragma omp parallel
    {
        float B;
        int i, id, nthrds;

        id = omp_get_thread_num();
        nthrds = omp_get_num_threads();

        for (i = id; i < niters; i += nthrds) {
            B = big_job(i);

            #pragma omp critical
            res += consume(B);
        }
    }

    std::cout << "Final result: " << res << std::endl;
    return 0;
}
