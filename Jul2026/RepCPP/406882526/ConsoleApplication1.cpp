#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

using namespace std;

int main()
{
    int size = 100; 
    vector<int> v1(size), v2(size), v3(size);

    auto start = std::chrono::system_clock::now();
#pragma omp parallel for 
    for (int i = 0; i < size; i++) {
        v1[i] = rand() % 20 + 1;
        v2[i] = rand() % 20 + 1;
    }

    auto end = std::chrono::system_clock::now();
    auto res = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    std::cout << "Time of parallel init of vectors: " << res.count() << "ns\n";

    start = std::chrono::system_clock::now();
#pragma omp parallel for 
    for (int i = 0; i < size; i++) {
        v3[i] = v1[i] + v2[i];
    }

    end = std::chrono::system_clock::now();
    res = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "Time of parallel add of vectors: " << res.count() << "ns\n";


    start = std::chrono::system_clock::now();
    for (int i = 0; i < size; i++) {
        v1[i] = rand() % 20 + 1;
        v2[i] = rand() % 20 + 1;
    }
    end = std::chrono::system_clock::now();
    res = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    std::cout << "Time of seq init of vectors: " << res.count() << "ns\n";

    start = std::chrono::system_clock::now();
    for (int i = 0; i < size; i++) {
        v3[i] = v1[i] + v2[i];
    }
    end = std::chrono::system_clock::now();
    res = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cout << "Time of seq add of vectors: " << res.count() << "ns\n";
}

