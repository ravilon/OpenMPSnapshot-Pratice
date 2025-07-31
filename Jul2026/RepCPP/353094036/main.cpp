/**
 * @file main.cpp
 * @class main
 * @author Liliana O'Sullivan
 * @brief An implementation of the Sieve of Eratosthenes using OpenMP.
 * @version 0.1
 * @date 12th February 2021
 * @copyright MIT License
 * 
 */

#include <iostream>
#include <chrono>
#include <omp.h>
#include <math.h>
using namespace std;

/**
 * @brief A sequencial implementation of the Sieve. Uses a stack based sieve. This segment-faults at a high value of N. Eg 1 Billion
 * 
 * @param n The value to calculate primes up to.
 * @return long Amount of twin primes found up to N.
 */
long sequential_stack(long long n)
{
    bool a[n];

    for (long long i = 0L; i < n; ++i)
        a[i] = true;

    for (long long i = 2; i * i <= n; ++i)
        if (a[i])
            for (long long j = i * i; j <= n; j += i)
                a[j] = false;

    long count = 0L;
    for (long long i = 3; i < n - 2; i+=2)
    {
        if (a[i] and a[i + 2])
        {
            cout << "(" << i << "," << i + 2 << ")" << endl;
            count++;
        }
    }
    cout << "\t\tSequential_stack: " << count << endl;
    return count;
}

/**
 * @brief A sequencial implementation of the Sieve. Uses a heap based sieve.
 * 
 * @param n The value to calculate primes up to.
 * @return long Amount of twin primes found up to N.
 */
long sequential_heap(long long n)
{
    bool *a = new bool[n];

    for (long long i = 0L; i < n; ++i)
        a[i] = true;

    for (long long i = 2; i * i <= n; ++i)
        if (a[i])
            for (long long j = i * 2; j <= n; j += i)
                a[j] = false;

    long count = 0L;
    for (long long i = 3; i < n - 2; i+=2)
    {
        if (a[i] and a[i + 2])
            count++;
    }
    cout << "\t\tSequential_heap: " << count << endl;
    delete[] a;
    return count;
}

/**
 * @brief A concurrent implementation of the Sieve. Uses a heap based sieve.
 * 
 * @param n The value to calculate primes up to.
 * @return long Amount of twin primes found up to N.
 */
long concurrent(long long n) 
{
    bool *a = new bool[n];

    // omp_set_num_threads(1);
    #pragma omp parallel for
    for (int i = 0; i < n; ++i)
        a[i] = true; // Assume all numbers prime by default

    const unsigned long n_sqrt = sqrt(n);

    #pragma omp parallel for schedule(dynamic) // Dynamic scheduling greatly increased performance
    for (long long i = 2; i <= n_sqrt; ++i)
        if (a[i]) // If its a prime, mark multiples as none-prime
            for (long long j = i * 2; j <= n; j += i)
                a[j] = false;

    unsigned long count = 0L;
    #pragma omp parallel for schedule(static)
    for (long long i = 3; i < n - 2; i+=2)
    {
        if (a[i] and a[i + 2])
        {
            #pragma omp critical
            count++;
        }
    }
    delete[] a;
    cout << "\t\tConcurrent: " << count << endl;
    return count;
}

int main()
{
    long long n = 500000000;
    // cin >> n;
    //sequential_heap(n);
    concurrent(n);
    return 0;
}