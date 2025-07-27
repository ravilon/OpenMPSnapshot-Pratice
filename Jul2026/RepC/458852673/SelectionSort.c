#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

void selection_sort(int *v, int n);

int main(int argc, char **argv){
	// omp_set_num_threads(2);
	int n = 10e2 * atoi(argv[1]);
	double start = 0, end = 0;
	int *list = (int*) malloc(n * sizeof(int));

	srand(time(NULL));

	for(i = 0; i < n; i++) list[i] = (rand() % (n));

	start = omp_get_wtime();

    selection_sort(list, n);

	end = omp_get_wtime();

	printf("\nSelectionSort\ntime: %f\nn: %d \n", end - start, n);

	return 0;
}

void selection_sort(int *v, int n){
	int i, j, min, tmp, min_local;

	for(i = 0; i < n - 1; i++){

		#pragma omp parallel private(j, min_local)
		{
			min = i;
			min_local = i;

			#pragma omp for
			for(j = i + 1; j < n; j++){
				if(v[j] < v[min_local])
					min_local = j;
			}

			#pragma omp critical
			if(v[min_local] < v[min]) min = min_local;
		}
		
		tmp = v[i];
		v[i] = v[min];
		v[min] = tmp;	
	}

}