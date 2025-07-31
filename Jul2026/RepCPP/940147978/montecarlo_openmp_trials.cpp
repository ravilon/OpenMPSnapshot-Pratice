#include <iostream>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <random>
#include <omp.h>

using namespace std;
using namespace std::chrono;

const long double TRUE_PI = 3.141592653589793238462643383279502884L;
const int NUM_THREADS = 4;

long double monteCarloPi(long long samples, long long& insideCircle) {
    insideCircle = 0;

    #pragma omp parallel num_threads(NUM_THREADS)
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

long double getSamplesForPrecision(long long terms) {
    long long samples = terms;
    long long insideCircle;
    long double piEstimate = monteCarloPi(samples, insideCircle);

    return piEstimate;
}

int main() {
    
    long long samples[] = { static_cast<long long>(pow(2,24)), 
                            static_cast<long long>(pow(2,26)), 
                            static_cast<long long>(pow(2,28)) };

    // long double precisions[] = {1e-5L, 1e-10L, 1e-15L, 1e-20L};
    // int precisionLevels[] = {5, 10, 15, 20};
    // long long samples[] = {100000, 10000000000, 1000000000000000, 100000000000000000000 };

    for (int i = 0; i < 4; i++) {
        auto start = high_resolution_clock::now();
        long double estimatedPi = getSamplesForPrecision(samples[i]);
        cout << "Samples: " << samples[i] << "\n";
        cout << "Estimated Pi: " << estimatedPi << "\n";
        auto end = high_resolution_clock::now();
        double duration = duration_cast<milliseconds>(end - start).count() / 1000.0;
        cout << "Error: " << fabsl(estimatedPi - TRUE_PI) << "\n";
        cout << "Total Execution Time: " << duration << " seconds\n\n";
        
    }
    return 0;
}
