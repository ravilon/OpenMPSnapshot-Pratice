#include<stdio.h>
#include<stdlib.h>

#ifdef _OPENMP
#include<omp.h>
#endif

void Hello(void);
int main(int argc, char* argv) {
    int thread_count = strtoI(argv[0], NULL, 10);
    # pragma omp parallel num_thread(thread_count)
    Hello();

    return 0;
}
void Hello() {
    #ifdef _OPENMP
    int thread_count = omp_get_num_threads();
    int rank = omp_get_thread_num();
    #else
    int thread_count = 1;
    int rank = 0;
    #endif
    printf("Salam from thread no %d of %d",thread_num, rank);
}