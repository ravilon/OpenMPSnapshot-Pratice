#include <iostream>
#include <omp.h>

int main()
{
	int a[10] = {50, 21, 42, 6, 1, 2, 8, 9, 10, 99};
	int b[10] = {9, 5, 1, 3, 9, 3, 8, 234, 0, -10};

	int min = INT32_MAX;
	int max = INT32_MIN;
	int threadsCount = 2;

	#pragma omp parallel shared(a, b) num_threads(threadsCount)
	{
		#pragma omp master
		{
			int size = sizeof(a) / sizeof(a[0]);
			for (int ind = 0; ind < size; ind += 1) {
				if (a[ind] < min) {
					min = a[ind];
				}
			}
			printf("Min value of array a = %d \n", min);
			printf("Thread №: %d \n\n", omp_get_thread_num());
		}

		#pragma omp single
		{
			int size = sizeof(b) / sizeof(b[0]);
			for (int ind = 0; ind < size; ind += 1) {
				if (b[ind] > max) {
					max = b[ind];
				}
			}
			printf("Max value of array b = %d \n", max);
			printf("Thread №: %d \n\n", omp_get_thread_num());
		}
	}

	printf("Done! min = %d, max = %d \n", min, max);
}
