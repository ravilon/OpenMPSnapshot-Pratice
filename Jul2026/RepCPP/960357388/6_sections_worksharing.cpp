// Worksharing - Sections (#pragma omp sections, #pragma omp section)

// Concept: Assigns different structured blocks of code (#pragma omp section) within a #pragma omp sections region to different threads in the team. Good for task parallelism where tasks are distinct.

#include <iostream>
#include <omp.h>
#include <thread> // For sleep
#include <chrono>

void taskA() {
    std::cout << "Task A starting on thread " << omp_get_thread_num() << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    std::cout << "Task A finished on thread " << omp_get_thread_num() << std::endl;
}

void taskB() {
    std::cout << "Task B starting on thread " << omp_get_thread_num() << std::endl;
     std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::cout << "Task B finished on thread " << omp_get_thread_num() << std::endl;
}
void taskC() {
     std::cout << "Task C starting on thread " << omp_get_thread_num() << std::endl;
     std::this_thread::sleep_for(std::chrono::milliseconds(150));
    std::cout << "Task C finished on thread " << omp_get_thread_num() << std::endl;
}


int main() {
     std::cout << "--- Sections Example ---" << std::endl;
     omp_set_num_threads(3); // Use at least 3 threads for this example

    #pragma omp parallel // Need a parallel region first
    {
        #pragma omp sections // Divide work within the region
        {
            #pragma omp section // First block of work
            {
                taskA();
            } // End section

            #pragma omp section // Second block of work
            {
                taskB();
            } // End section

            #pragma omp section // Third block of work
            {
                 taskC();
            } // End section

        } // End sections - Implicit barrier here unless 'nowait' is added
    } // End parallel region

    std::cout << "Sections finished." << std::endl;
    return 0;
}