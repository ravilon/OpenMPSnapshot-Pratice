/*
#include <stdio.h>
#include <omp.h>
#include <time.h>

int main(){
    int parallel_sum,serial_sum=0;

    double start_parallel,end_parallel;
    clock_t serial_start, serial_end;

    start_parallel=omp_get_wtime();

        #pragma omp parallel for
        for(int i=1;i<1000;i++) {
            parallel_sum+=i;
        }
        printf("Sum is %d",parallel_sum);

    end_parallel=omp_get_wtime();
    printf("\n-----------------------------\n");
    printf("Parallel Execution time = %f",end_parallel-start_parallel);
    printf("\n-----------------------------\n");

    serial_start=clock();

    for(int i=1;i<1000;i++) {
            serial_sum+=i;
        }
    printf("Sum is %d",serial_sum);

    serial_end=clock();
    double serial_execution_time=(double)(serial_end-serial_start)/CLOCKS_PER_SEC;

    printf("\n-----------------------------\n");
    printf("Serial Execution time = %f",serial_execution_time);
    printf("\n-----------------------------\n");


}
*/

#include <stdio.h>
#include <omp.h>
#include <time.h>

int main() {
    int parallel_sum = 0, serial_sum = 0;
    const int N=10000000;
    double start_parallel, end_parallel;
    clock_t serial_start, serial_end;

    start_parallel = omp_get_wtime();

    #pragma omp parallel for reduction(+:parallel_sum)
    for (int i = 1; i < N; i++) {
        parallel_sum += i;
    }

    end_parallel = omp_get_wtime();

    printf("Sum is %d\n", parallel_sum);

    printf("-----------------------------\n");
    printf("Parallel Execution time = %f\n", end_parallel - start_parallel);
    printf("-----------------------------\n");

    serial_start = clock();

    for (int i = 1; i < N; i++) {
        serial_sum += i;
    }

    printf("Sum is %d\n", serial_sum);

    serial_end = clock();
    double serial_execution_time = (double)(serial_end - serial_start) / CLOCKS_PER_SEC;

    printf("-----------------------------\n");
    printf("Serial Execution time = %f\n", serial_execution_time);
    printf("-----------------------------\n");

    return 0;
}
