// Concept: Controls how variables defined outside the parallel region are treated inside it by different threads. This is CRITICAL for correctness.

// shared(var1, var2): The variable is shared by all threads (default for most variables defined outside). Access needs synchronization if written to.
// private(var1, var2): Each thread gets its own uninitialized copy of the variable.
// firstprivate(var1, var2): Each thread gets its own copy, initialized with the value the variable had before the parallel region/loop.
// lastprivate(var): Like private, but the value from the thread that executes the last iteration (for loops) or section is copied back to the original variable outside.
// default(shared | none): Sets the default scoping. none forces you to explicitly scope almost all variables, which is safer.

#include <iostream>
#include <vector>
#include <omp.h>

int main() {
    std::cout << "--- Data Scoping Example ---" << std::endl;
    int shared_var = 100;
    int firstpriv_var = 200;
    int private_var_outside = 300; // Just to show it's distinct

    omp_set_num_threads(3);

    // Use default(none) to force explicit scoping - good practice!
    #pragma omp parallel default(none) \
            shared(shared_var, std::cout) \
            firstprivate(firstpriv_var) \
            private(private_var_outside) // Each thread gets uninitialized copy
    {
        int thread_id = omp_get_thread_num();
        // private_var_outside is uninitialized here for each thread
        private_var_outside = thread_id * 10; // Initialize this thread's copy

        // Modify this thread's copy of firstpriv_var
        firstpriv_var += thread_id;

        // Access the single shared variable (potential race if written without sync)
        // shared_var += thread_id; // UNSAFE without synchronization!

        #pragma omp critical
        {
            std::cout << "Thread " << thread_id
                      << ": shared_var = " << shared_var // Reads the shared value
                      << ", firstpriv_var = " << firstpriv_var // Reads thread's initialized copy
                      << ", private_var_outside = " << private_var_outside // Reads thread's copy
                      << std::endl;
        }
    } // End parallel region

    std::cout << "After parallel region:" << std::endl;
    std::cout << "shared_var = " << shared_var << " (unchanged by threads in this safe example)" << std::endl;
    std::cout << "firstpriv_var = " << firstpriv_var << " (original value unchanged)" << std::endl;
    std::cout << "private_var_outside = " << private_var_outside << " (original value unchanged)" << std::endl;

    return 0;
}