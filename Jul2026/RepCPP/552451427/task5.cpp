#include <iostream>
#include <omp.h>
#include <algorithm>


void print_array(int arr[], int columns)
{
    for (int i = 0; i < columns; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

void print_2array(int arr[][8], int rows)
{
    for (int i = 0; i < rows; i++) {
        print_array(arr[i], 8);
    }
    printf("\n");
}

int main()
{
	const int rows = 6;
	const int columns = 8;
	int maxValue = 1000;

	int array[rows][columns];

	srand(time(NULL));

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++) {
			array[i][j] = rand() % maxValue;
		}
	}

	printf("Matrix: \n");
	print_2array(array, rows);

	#pragma omp parallel sections
	{
		#pragma omp section 
		{
			int sum = 0;
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < columns; j++) {
					sum += array[i][j];
				}
			}
			int avg = sum / (rows * columns);

			printf("Average value = %d; from thread №%d \n", avg, omp_get_thread_num());
		}

		#pragma omp section 
		{
			int max = INT32_MIN;
			int min = INT32_MAX;

			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < columns; j++) {
					if (array[i][j] > max) {
						max = array[i][j];
					}

					if (array[i][j] < min) {
						min = array[i][j];
					}
				}
			}

			printf("Min = %d, max = %d; from thread №%d \n", min, max, omp_get_thread_num());
		}

		#pragma omp section 
		{
			int count = 0;
			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < columns; j++) {
					if (array[i][j] % 3 == 0) {
						count++;
					}
				}
			}

			printf("Count = %d; from thread №%d\n", count, omp_get_thread_num());
		}
	}
	printf("\n");
}