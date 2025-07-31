#include <omp.h>

int main() {
    int var1, var2, var3;

    // Sequential code
    // ...

    // The parallel section begins.
    // The master thread expands to a set of threads.
    // The scope of variables is specified.
    #pragma omp parallel private(var1, var2) shared(var3)
    {
        // Parallel section executed by all threads
        // ...
        // Every thread joins the master thread
    }
    // Sequential code continues in the master thread
    
    return 0;
}
