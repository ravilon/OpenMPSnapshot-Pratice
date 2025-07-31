#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <random>
#include <omp.h>

using namespace std;
using namespace std::chrono;

const long double TRUE_PI = 3.141592653589793238462643383279502884L;
// const int NUM_THREADS = 24;

long double monteCarloPi(long long samples, long long& insideCircle) {
    insideCircle = 0;

    // #pragma omp parallel num_threads(NUM_THREADS)
    #pragma omp parallel
    {
        random_device rd;
        mt19937 gen(rd() + omp_get_thread_num());
        uniform_real_distribution<long double> dist(0.0L, 1.0L);

        long long localInside = 0;

        #pragma omp for
        for (long long i = 0; i < samples; i++) {
            long double x = dist(gen);
            long double y = dist(gen);
            if (x * x + y * y <= 1) localInside++;
        }

        #pragma omp atomic
        insideCircle += localInside;
    }
    return (4.0L * insideCircle) / samples;
}

long long getSamplesForPrecision(long double requiredPrecision, long long terms) {
    long long samples = terms;
    long long insideCircle;
    long double piEstimate, error;

    while (true) {
        piEstimate = monteCarloPi(samples, insideCircle);
        error = fabsl(piEstimate - TRUE_PI);

        if (error <= requiredPrecision) {
            return samples;
        }
        samples *= 2;
    }
}

int main() {
    

    long double precisions[] = {1e-3L, 1e-5L, 1e-10L, 1e-15L, 1e-20L};
    int precisionLevels[] = {3, 5, 10, 15, 20};
    unsigned long long samples[] = {10000, 1000000000, 100000000000000, 10000000000000000000 };

    for (int i = 0; i < 5; i++) {
        auto start = high_resolution_clock::now();
        long long requiredSamples = getSamplesForPrecision(precisions[i], samples[i]);
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
