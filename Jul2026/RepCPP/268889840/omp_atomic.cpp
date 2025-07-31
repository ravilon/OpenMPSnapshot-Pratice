#include <iostream>
#include <omp.h>
#include <unistd.h> // usleep function
#include <vector>

int main()
{
    int n_threads, thread_id;

    int size = 10;
    std::vector<double> a(size), b(size);
    std::fill(a.begin(), a.end(), 2);
    std::fill(b.begin(), b.end(), 3);
    double dot_prod = 0.0;

    // Fork a team of threads
    // Note "shared dot_prod" variable
    #pragma omp parallel private(n_threads, thread_id) shared(dot_prod)
    {
        #pragma omp single
        {
            // Print number of threads only in one available thread
            n_threads = omp_get_num_threads();
            std::cout << "Parallel Region Start ..." << std::endl;
            std::cout << "Number of threads = " << n_threads << std::endl;
        }
        thread_id = omp_get_thread_num();
        // Delay each thread to avoid race condition on std::cout
        usleep(5000 * thread_id);
        #pragma omp for
        for (int i = 0; i < size; i++)
        {
            #pragma omp atomic
            dot_prod += a[i]*b[i];
            std::cout << "Thread " << thread_id << " executes iteration " << i << "."<< std::endl;
        }
        #pragma omp single
        std::cout << "Parallel Region End ..." << std::endl;
    }
    std::cout << "Dot Product: " << dot_prod << " (expected 60)."<< std::endl;
}
