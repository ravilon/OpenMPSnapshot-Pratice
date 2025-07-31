#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 32

int x, res[N], add[N], sum[N];

void seq() {
	for (int i = 0; i < N - 2; i++)
	{
		int j = N - 1 - i;
		sum[j] = sum[j - 1] + res[j] + add[j];
	}
}

void par() {
	int sum2[N];

#pragma omp parallel for
	for (int i = 0; i < N; i++)
	{
		sum2[i] = sum[i];
	}

	int j;

#pragma omp parallel for
	for (int i = 0; i < N - 2; i++)
	{
		j = N - 1 - i;
		sum[j] = sum2[j - 1] + res[j] + add[j];
	}
}

void print() {
	for (size_t i = 0; i < N; i++)
	{
		printf("%d ", res[i]);
	}
	printf("\n");
	for (size_t i = 0; i < N; i++)
	{
		printf("%d ", add[i]);
	}
	printf("\n");
	for (size_t i = 0; i < N; i++)
	{
		printf("%d ", sum[i]);
	}
	printf("\n");
	printf("%d ", x);
	printf("\n");
}

void init() {
	omp_set_num_threads(8);
	printf("-%d-\n", omp_get_num_procs());

	x = 0;

	for (size_t i = 0; i < N; i++)
	{
		res[i] = add[i] = sum[i] = i + 1;
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
