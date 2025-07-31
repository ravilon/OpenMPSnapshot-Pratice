#include <iostream>
#include <omp.h>
#include <unistd.h> // usleep function
#include <vector>

int main()
{
    int n_threads, thread_id;

    double value = 5.0;
    int size = 10;
    std::vector<double> a(size), b(size), c(size);
    std::fill(a.begin(), a.end(), 0.6*value);
    std::fill(b.begin(), b.end(), 0.4*value);
    std::fill(c.begin(), c.end(), 0.0);

    // Fork a team of threads
    #pragma omp parallel private(n_threads, thread_id)
    {
        #pragma omp master
        {
            // Ensures printing the number of threads only in the main (or master) thread
            thread_id = omp_get_thread_num();
            n_threads = omp_get_num_threads();
            std::cout << "Parallel Region Start ... (thread #" << thread_id << ")" << std::endl;
            std::cout << "Number of threads = " << n_threads << std::endl;
        }
        #pragma omp barrier // The "master" constructor does not have implicit barrier.
                            // Comment the barrier line to see the difference.

        thread_id = omp_get_thread_num();
        // Delay each thread to avoid race condition on std::cout
        // usleep(5000 * thread_id);
        #pragma omp for
        for (int i = 0; i < size; i++)
        {
            c[i] = a[i]+b[i];
            std::cout << "Thread " << thread_id << " executes iteration " << i << "."<< std::endl;
        }
    }
}