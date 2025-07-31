#include <iostream>
#include <omp.h>

int main()
{
    int a = 0;
    int b = 0;

    int threadsCount = 2;

	std::cout << "1st area \n";
	printf("Before entering first area. Threads count = %d, a = %d, b = %d\n", threadsCount, a, b);
	#pragma omp parallel num_threads(threadsCount) private(a) firstprivate(b)
	{
        a = 0;
        b = 0;
        #pragma omp critical (summ) 
        { 
            int thread_id = omp_get_thread_num();

            a += thread_id;
            b += thread_id;
            printf("       Thread № = %d, a = %d, b = %d\n", thread_id, a, b);
        }
    }
	printf("After exiting first area. Threads count = %d, a = %d, b = %d\n", threadsCount, a, b);

	std::cout << "\n2nd area\n";

    threadsCount = 4;
	printf("Before entering second area. Threads count = %d, a = %d, b = %d\n", threadsCount, a, b);
	#pragma omp parallel num_threads(threadsCount) shared(a) private(b)
	{
        a = 0;
        b = 0;
       #pragma omp critical (minus) 
        { 
            int thread_id = omp_get_thread_num();

            a -= thread_id;
            b -= thread_id;
            printf("       Thread № = %d, a = %d, b = %d\n", thread_id, a, b);
        }

	}
	printf("After exiting second area. Threads count = %d, a = %d, b = %d\n", threadsCount, a, b);
}
