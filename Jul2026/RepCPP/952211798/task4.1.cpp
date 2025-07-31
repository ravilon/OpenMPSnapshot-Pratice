#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace std::chrono;

int main() {
    const size_t N = 100000000;
    vector<double> A(N, 1.0), B(N, 2.0), C(N, 0.0);

    cout << "THREADS, TEMPO (ms)" << endl;
    
    for (int num_threads = 1; num_threads <= 32; num_threads = num_threads + 1) {
        omp_set_num_threads(num_threads);
        
        auto start = high_resolution_clock::now();
        #pragma omp parallel for
        for (size_t i = 0; i < N; i++) {
            C[i] = A[i] + B[i];
        }
        auto end = high_resolution_clock::now();
        double duration = duration_cast<milliseconds>(end - start).count();
        
        cout << num_threads << ", " << duration << endl;
    }
    return 0;
}