#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace std::chrono;

const long double TRUE_PI = 3.141592653589793238462643383279502884L;
const int NUM_THREADS = 124;

long double monteCarloPi(long long samples, long long& insideCircle) {
    insideCircle = 0;

    #pragma omp parallel num_threads(NUM_THREADS) reduction(+:insideCircle)
    {
        unsigned int seed = 785 + omp_get_thread_num(); // Unique seed per thread
        long long threadInside = 0; // Private variable for each thread

        #pragma omp for
        for (long long i = 0; i < samples; i++) { 
            long double x = static_cast<long double>(rand_r(&seed)) / RAND_MAX;
            long double y = static_cast<long double>(rand_r(&seed)) / RAND_MAX;
            if (x * x + y * y <= 1) threadInside++;
        }

        insideCircle += threadInside; // Reduction sums up all thread results
    }

    return (4.0L * insideCircle) / samples;
}

long long getSamplesForPrecision(long double requiredPrecision) {
    long long samples = 100000;
    long long insideCircle;
    long double piEstimate, error;

    while (true) {
        piEstimate = monteCarloPi(samples, insideCircle);
        error = fabsl(piEstimate - TRUE_PI);

        if (error <= requiredPrecision) {
            return samples;
        }
        samples = static_cast<long long>(samples * 1.5);
    }
}

int main() {
    long double precisions[] = {1e-3L, 1e-5L, 1e-10L, 1e-15L, 1e-20L};
    int precisionLevels[] = {3, 5, 10, 15, 20};

    for (int i = 0; i < 5; i++) {
        auto start = high_resolution_clock::now();
        long long requiredSamples = getSamplesForPrecision(precisions[i]);
        long long insideCircle;
        long double estimatedPi = monteCarloPi(requiredSamples, insideCircle);

        cout.precision(precisionLevels[i] + 2);
        cout << "Precision: " << precisionLevels[i] << " decimal places\n";
        cout << "Estimated Pi: " << estimatedPi << "\n";
        cout << "Required Samples: " << requiredSamples << "\n";
        auto end = high_resolution_clock::now();
        double duration = duration_cast<milliseconds>(end - start).count() / 1000.0;
        cout << "Total Execution Time: " << duration << " seconds\n\n";
    }

    return 0;
}
