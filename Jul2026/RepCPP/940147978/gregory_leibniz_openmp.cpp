#include <iostream>
#include <cmath>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace std::chrono;

const int NUM_THREADS = 24;
const long double TRUE_PI = 3.141592653589793238462643383279502884L;

// Function to compute Pi using OpenMP
long double compute_pi(long long terms) {
    long double piSum = 0.0L;

    #pragma omp parallel for num_threads(NUM_THREADS) reduction(+:piSum)
    for (long long k = 0; k < terms; k++) {
        long double term = (k % 2 == 0 ? 1.0L : -1.0L) / (2 * k + 1);
        piSum += term;
    }

    return 4.0L * piSum;
}

// Determine number of terms for given precision
long long get_terms_for_precision(long double requiredPrecision, long long sample) {
    long long terms = sample;
    long double estimatedPi, error;

    while (true) {
        estimatedPi = compute_pi(terms);
        error = fabsl(estimatedPi - TRUE_PI);
        if (error <= requiredPrecision) return terms;
        terms *= 2;
    }
}

int main() {
    long double precisions[] = {1e-3L, 1e-5L, 1e-10L, 1e-15L, 1e-20L};
    int precisionLevels[] = {3, 5, 10, 15, 20};
    long long samples[] = {100000, 10000000000, 1000000000000000, 100000000000000000000 };

    for (int i = 0; i < 5; i++) {
        auto start = high_resolution_clock::now();
        long long requiredTerms = get_terms_for_precision(precisions[i], samples[i]);
        long double estimatedPi = compute_pi(requiredTerms);

        cout.precision(precisionLevels[i] + 2);
        cout << "Precision: " << precisionLevels[i] << " decimal places\n";
        cout << "Estimated Pi: " << estimatedPi << "\n";
        cout << "Required Terms: " << requiredTerms << "\n";

        auto end = high_resolution_clock::now();
        double duration = duration_cast<milliseconds>(end - start).count() / 1000.0;
        cout << "Total Execution Time: " << duration << " seconds" << "\n\n";
    }
    return 0;
}
