#include <math.h>
#include <stdio.h>
#define N 5


void main()
{
	int i; double A[N] = { 1,2,3,4,5 }, B[N] = { 1,2,3,4,5 }, C[N] = { 1,2,3,4,5 }, D[N];
	const double c = 5;
	const double x = 2;
	double y;

#pragma omp parallel for lastprivate(y)
	for (i = 0; i < N; i++)
	{
		y = sqrt(A[i]);
		D[i] = y + A[i] / (x * x);
	}

	for (int i = 0; i < N; i++)
	{
		printf("%lf ", D[i]);
	}
	printf("\n%lf", y);
}
