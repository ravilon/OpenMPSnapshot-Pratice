#include <iostream>
#include <omp.h>

int main() {

    printf("Executing the section with 3 threads: \n");
	omp_set_num_threads(3);

	if (omp_get_max_threads() == 1) {
        return 0;
    }
	#pragma omp parallel if(omp_get_max_threads() > 1)
	{
		printf("Threads count = %d; Thread № = %d\n", omp_get_num_threads(), omp_get_thread_num());
	}


    printf("Executing the section with 1 thread: \n");
	omp_set_num_threads(1);

	if (omp_get_max_threads() == 1) {
        return 0;
    }
	#pragma omp parallel if(omp_get_max_threads() > 1)
	{
		printf("Threads count = %d; Thread № = %d\n", omp_get_num_threads(), omp_get_thread_num());
	}
}