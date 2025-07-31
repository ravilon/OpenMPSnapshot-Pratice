#include <omp.h>
#include <cstdio>

void pooh(int ID, double A[]) {
    // Example function definition (modify as needed)
    printf("Thread %d is executing pooh()\n", ID);
}

int main() {
    double A[1000];
    omp_set_num_threads(4);
    
    #pragma omp parallel
    {
        int ID = omp_get_thread_num();  // Fixed typo
        pooh(ID, A);  // Ensure pooh() is defined
    }

    printf("All done\n");  // Fixed typo in message
    return 0;
}
