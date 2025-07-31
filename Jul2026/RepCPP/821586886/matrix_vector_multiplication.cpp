#include <chrono>
#include <iostream>
#include <vector>
#include <omp.h>

// this is simplified sequential BLAS DGBMV
void dmv(int n, int m, const double* A, const double* x , double* y) {
    for (int i = 0; i < n; ++i) {
        const double* Ai = A + i * m;
        for (int j = 0; j < m; ++j) {
            y[i] += (Ai[j] * x[j]);
        }
    }
} // dmv

// this is the most basic parallelization of DGBMV using OpenMP
void omp_dmv(int n, int m, const double* A, const double* x , double* y) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        const double* Ai = A + i * m;
        for (int j = 0; j < m; ++j) {
            y[i] += (Ai[j] * x[j]);
        }
    }
} // omp_dmv


int main() {
    int n, m;
    int n_threads;

    std::cout << "Number of Matrix/Vector Order:";
    std::cin >> n;
    
    m = n;

    if ((n < 2) || (m < 2)) {
        std::cout << "error: incorrect Order of the Problem" << std::endl;
        return -1;
    }

    unsigned long long int Asz = (1ULL * n) * m;

    std::vector<double> A(Asz);
    std::vector<double> x(m);
    std::vector<double> y(n);

    std::cout << "Number of Threads:";
    std::cin >> n_threads;
    omp_set_num_threads(n_threads);
    auto t0 = std::chrono::steady_clock::now();

    // call sequential or parallel solver (choose one)
    //dmv(n, m, A.data(), x.data(), y.data());
    omp_dmv(n, m, A.data(), x.data(), y.data());

    auto t1 = std::chrono::steady_clock::now();

    // report time in seconds
    std::chrono::duration<double> dT = t1 - t0;
    std::cout << dT.count() << std::endl;

    return 0;
} // main
