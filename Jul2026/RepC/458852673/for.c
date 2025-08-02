#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

void printArray(int *array, int n);

int main(int argc, char **argv){
    srand(time(NULL));

    if(!argv[1]){
        printf("Enter the array size when calling the command (e.g ./file.out n)\n");
        exit(1);
    }

    int n = atoi(argv[1]), i = 0, j = 0;

    int *a = (int *)malloc(n * sizeof(int));
    int *b = (int *)malloc(n * sizeof(int));

    double start = 0, end = 0;

	start = omp_get_wtime();
    #pragma omp parallel shared(n,a,b) firstprivate(i,j)
    {
        #pragma omp for
        for (i=0; i<n; i++){
            if(j == 0) printf("[thread:%d start:%d] START a\n", omp_get_thread_num(), i);
            j = i;
            a[i] = i;
        }
        printf("[thread:%d end:%d] END a\n", omp_get_thread_num(), j);

        j = 0;
        #pragma omp for
        for (i=0; i<n; i++){
            if(j == 0) printf("[thread:%d start:%d] START b\n", omp_get_thread_num(), i);
            j++;
            b[i] = 2 * a[i];
        }
        printf("[thread:%d end:%d] END b\n", omp_get_thread_num(), j);

    }
    end = omp_get_wtime();
    // printf("b:\n");
    // printArray(b, n);

    printf("%f\n", end - start);

    free(a);
    free(b);

    return 0;
}

void printArray(int *array, int n){
    for(int i = 0; i < n; i++){
        printf("%d ", array[i]);
        printf("\n");
    }
}