#include <iostream>
#include <omp.h>
#include <chrono>

using namespace std;

bool is_prime(int num) {
    if (num <= 1) {
        return false;
    }

    for (int i = 2; i * i <= num; ++i) {
        if (num % i == 0) {
            return false;
        }
    }

    return true;
}

int main() {
    int n = 100000;
    int count = 0;
    printf("Done by 2021BCS0084 Dhivyesh RK\n");
	auto start_time = chrono::high_resolution_clock::now();
	for (int i = 1; i <= n; ++i) {
        if (is_prime(i)) {
            ++count;
        }
    }
	auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    printf("Total prime numbers between 1 and %d without parallel: %d\n", n, count);
    printf("Time taken without parallel: %ld milliseconds\n\n", duration.count());
	count = 0;
    auto start_time2 = chrono::high_resolution_clock::now();

    #pragma omp parallel for reduction(+:count)
    for (int i = 1; i <= n; ++i) {
        if (is_prime(i)) {
            ++count;
        }
    }

    auto end_time2 = chrono::high_resolution_clock::now();
    auto duration2 = chrono::duration_cast<chrono::milliseconds>(end_time2 - start_time2);
    printf("Total prime numbers between 1 and %d using parallel: %d\n", n, count);
    printf("Time taken using parallel processing: %ld milliseconds\n", duration2.count());
    printf("\n Parallel processing is : %lf percent faster\n", float(float(duration.count())/float(duration2.count()))*100);
    
  
	

    return 0;
}

