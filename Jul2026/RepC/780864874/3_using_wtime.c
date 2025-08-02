
#include <stdio.h>
#include <omp.h>

int main(){
    double start_time,end_time;

    // Starts measuring time
    start_time=omp_get_wtime();

    #pragma omp parallel
    {
        printf("Hello There! We are measuring time to execute this parallel region!\n");
    }

    end_time=omp_get_wtime();
    // Adds a time stamp

    double elapsed_time=end_time-start_time; // Gets the time difference
    printf("\n----------------------------\n");
    printf("You have spent %f seconds to execute parallel region",elapsed_time);
    printf("\n----------------------------\n");
}
