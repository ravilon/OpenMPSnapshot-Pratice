#include <iostream>
#include <omp.h>
#include <fstream>
#include <vector>
#include <chrono>


const long num_steps = 100000000;
const int block_size = 1308080;
double step;


int main() 
{
    // Different number of threads to test
    std::vector<int> threads = {1, 2, 4, 8, 12, 16, 32, 64}; 
    std::ofstream resultsFile("Threaded_Pi_Calc_OpenMP_results.txt");
    resultsFile << "Threads, Time taken (s)" << std::endl;

    for (int numThreads : threads) 
    {
        double pi, sum = 0.0;
        step = 1.0 / (double)num_steps;

        // Set the number of threads
        omp_set_num_threads(numThreads);

        // Start timing
        std::cout << "Starting calculation with " << numThreads << " threads...\n";
        auto startTime = std::chrono::high_resolution_clock::now();

        // Perform the computation
        #pragma omp parallel for schedule(dynamic, block_size) reduction(+:sum)
        for (long i = 0; i < num_steps; i++) 
        {
            double x = (i + 0.5) * step;
            sum += 4.0 / (1.0 + x * x);
        }

        // End timing
        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = endTime - startTime;

        pi = step * sum;

        std::cout << "Threads: " << numThreads << ", Time taken: " << duration.count() << " s, Calculated Pi: " << pi << "\n";
        resultsFile << numThreads << ", " << duration.count() << std::endl;
    }

    resultsFile.close();
    return 0;
}
