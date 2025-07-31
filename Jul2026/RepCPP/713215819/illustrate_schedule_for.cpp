#include <stdio.h>
#include <omp.h>
#include <chrono>

int main() {
    int N = 100; 	
    int sum = 0;
    int i;

    printf("Name: Dhivyesh R K\nRoll No.:2021BCS0084\n");
    // Serial version
    printf("Serial Version:\n");
    auto serial_start = std::chrono::high_resolution_clock::now();
    for (i = 1; i <= N; i++) {
        sum += i;
    }
    auto serial_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> serial_duration = serial_end - serial_start;
    printf("Sum of first %d natural numbers: %d\n", N, sum);
    printf("Serial execution time: %f seconds\n\n", serial_duration.count());

    //Parallel Version
    printf("Parallel Version:\n");

    int sum_parallel = 0;
    auto parallel_start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for reduction(+:sum_parallel)
    for (i = 1; i <= N; i++) {
        sum_parallel += i;
    }

    auto parallel_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> parallel_duration = parallel_end - parallel_start;
    printf("Sum of first %d natural numbers (parallel with reduction): %d\n", N, sum_parallel);
    printf("Parallel execution time: %f seconds\n\n", parallel_duration.count());

    // Parallel version with different schedule clauses
    printf("Parallel Versions with different types:\n");

    // Schedule (static)
    int sum_static = 0;
    auto static_start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(static) reduction(+:sum_static)
    for (i = 1; i <= N; i++) {
        sum_static += i;
    }
    auto static_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> static_duration = static_end - static_start;
    printf("Sum of first %d natural numbers (static schedule): %d\n", N, sum_static);
    printf("Static execution time: %f seconds\n\n", static_duration.count());

    // Schedule (static, C)
    int sum_static_c = 0;
    auto static_c_start = std::chrono::high_resolution_clock::now();
    int chunk_size = 10; // Adjust C as needed

    #pragma omp parallel for schedule(static, chunk_size) reduction(+:sum_static_c)
    for (i = 1; i <= N; i++) {
        sum_static_c += i;
    }
    auto static_c_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> static_c_duration = static_c_end - static_c_start;
    printf("Sum of first %d natural numbers (static schedule with chunk size %d): %d\n", N, chunk_size, sum_static_c);
    printf("Static with chunk execution time: %f seconds\n\n", static_c_duration.count());

    // Schedule (dynamic)
    int sum_dynamic = 0;
    auto dynamic_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(dynamic) reduction(+:sum_dynamic)
    for (i = 1; i <= N; i++) {
        sum_dynamic += i;
    }
    auto dynamic_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dynamic_duration = dynamic_end - dynamic_start;
    printf("Sum of first %d natural numbers (dynamic schedule): %d\n", N, sum_dynamic);
    printf("Dynamic execution time: %f seconds\n\n", dynamic_duration.count());

    // Schedule (dynamic, C)
    int sum_dynamic_c = 0;
    auto dynamic_c_start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(dynamic, chunk_size) reduction(+:sum_dynamic_c)
    for (i = 1; i <= N; i++) {
        sum_dynamic_c += i;
    }
    auto dynamic_c_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dynamic_c_duration = dynamic_c_end - dynamic_c_start;
    printf("Sum of first %d natural numbers (dynamic schedule with chunk size %d): %d\n", N, chunk_size,
           sum_dynamic_c);
    printf("Dynamic with chunk execution time: %f   seconds\n\n", dynamic_c_duration.count());

    // Schedule (guided)
    int sum_guided = 0;
    auto guided_start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(guided) reduction(+:sum_guided)
    for (i = 1; i <= N; i++) {
        sum_guided += i;
    }
    auto guided_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> guided_duration = guided_end - guided_start;
    printf("Sum of first %d natural numbers (guided schedule): %d\n", N, sum_guided);
    printf("Guided execution time: %f seconds\n\n", guided_duration.count());

    // Schedule (guided, C)
    int sum_guided_c = 0;
    auto guided_c_start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(guided, chunk_size) reduction(+:sum_guided_c)
    for (i = 1; i <= N; i++) {
        sum_guided_c += i;
    }
    auto guided_c_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> guided_c_duration = guided_c_end - guided_c_start;
    printf("Sum of first %d natural numbers (guided schedule with chunk size %d): %d\n", N, chunk_size, sum_guided_c);
    printf("Guided with chunk execution time: %f seconds\n\n", guided_c_duration.count());

    return 0;
}

