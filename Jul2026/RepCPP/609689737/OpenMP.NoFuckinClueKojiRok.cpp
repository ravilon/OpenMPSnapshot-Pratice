#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 4

int h, a[N][N];

void seq() {
	for (int i = 2; i < N; i++)
	{
		for (int j = 0; j < N;j++) {
			h += a[i][j];
			a[i][j] = a[i - 2][j];
		}
	}
}

void par() {
#pragma omp parallel for reduction (+:h)
	for (int j = 0; j < N; j++)
	{
		for (int i = 2; i < N; i++)
		{
			h += a[i][j];
			a[i][j] = a[i-2][j];
		}
	}
}

void print() {
	for (size_t i = 0; i < N; i++)
	{
		for (size_t j = 0; j < N; j++)
		{
			printf("%d ", a[i][j]);

		}
		printf("\n");
	}

	printf("\n");
	printf("%d ", h);
	printf("\n");
}

void init() {
	omp_set_num_threads(8);
	printf("-%d-\n", omp_get_num_procs());

	h = 0;

	for (size_t i = 0; i < N; i++)
	{
		for (size_t j = 0; j < N; j++)
		{
			a[i][j] = i * N + j + 1;
		}
	}
}

int main() {
	init();

	double t = omp_get_wtime();
	seq();
	print();

	init();

	printf("%lf\n", omp_get_wtime() - t);
	t = omp_get_wtime();
	par();
	printf("---\n");
	print();

	printf("%lf\n", omp_get_wtime() - t);
	return 0;
}
