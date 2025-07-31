#include <iostream>
#include <omp.h>
#include <unistd.h> // usleep function
#include <vector>

int Number = 5;
#pragma omp threadprivate(Number)

int main()
{
    int n_threads, thread_id;

    std::cout << ">> Initial value of Number = " << Number << std::endl;

    // Fork a team of threads
    #pragma omp parallel private(n_threads, thread_id)
    {
        thread_id = omp_get_thread_num();
        #pragma omp master
        {
            // Ensures printing the number of threads only in the main (or master) thread
            n_threads = omp_get_num_threads();
            std::cout << "Parallel Region Start ... (thread #" << thread_id << ")" << std::endl;
            std::cout << "Number of threads = " << n_threads << std::endl;
        }
        #pragma omp barrier // The "master" constructor does not have implicit barrier.
                            // Comment the barrier line to see the difference.
        // Delay each thread to avoid race condition on std::cout
        usleep(5000 * thread_id);
        // Change the value of Number
        Number += thread_id + 1;
        std::cout << "Number = " << Number << " (at Thread #" << thread_id << ")." << std::endl;
    }
    std::cout << "Parallel Region End ... (thread #" << thread_id << ")" << std::endl;
    // The value added at Thread master persists outside of paralell region
    std::cout << ">> Final value of Number = " << Number << std::endl;
}
