#include <iostream>
#include <chrono>
#include <omp.h>
#define N 10

using namespace std; 

int main()
{
    int m1[N][N], m2[N][N], m3[N][N];

    auto start = std::chrono::system_clock::now();
#pragma omp parallel for 
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            m1[i][j] = rand() % 20 + 1;
            m2[i][j] = rand() % 20 + 1;
        }
    }
    auto end = std::chrono::system_clock::now();
    auto res = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "Time of parallel init of matrices: " << res.count() << "ns\n";

    start = std::chrono::system_clock::now();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            m3[i][j] = 0;
            for (int k = 0; k < N; k++) {
                m3[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }

    end = std::chrono::system_clock::now();
    res = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "Time of parallel mult of matrices: " << res.count() << "ns\n";


    start = std::chrono::system_clock::now();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            m1[i][j] = rand() % 20 + 1;
            m2[i][j] = rand() % 20 + 1;
        }
    }
    end = std::chrono::system_clock::now();
    res = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "Time of seq init of matrices: " << res.count() << "ns\n";

    start = std::chrono::system_clock::now();
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            m3[i][j] = 0;
            for (int k = 0; k < N; k++) {
                m3[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }

    end = std::chrono::system_clock::now();
    res = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "Time of seq mult of matrices: " << res.count() << "ns\n";
}

