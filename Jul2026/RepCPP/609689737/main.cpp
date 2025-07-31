#include <stdio.h>
int x = 0;
#define N 10
#define M 15

int a[N][M]{}, b[N][M]{};

void seq() {
	for (int i = 2; i < N; i++)
	{
		for (int j = 0;j < M;j++) {
			x = x + a[i][j];
			b[i][j] = 4 * b[i - 2][j];
		}
	}
}

void par() {
	int i = 0, j = 0;
#pragma omp parallel for reduction(+:x) private(i)
	for (j = 0; j < M; j++)
	{
		for (i = 2; i < N; i++)
		{
			x += a[i][j];
			b[i][j] = 4 * b[i - 2][j];
		}
	}
}


void reset() {
	x = 0;

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < M; j++)
		{
			a[i][j] = b[i][j] = 1;
		}
	}
}

void print() {
	printf("x=%d\n", x);
	printf("a:\n");
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < M; j++)
		{
			printf("%d ", a[i][j]);
		}
		printf("\n");
	}
	printf("b:\n");
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < M; j++)
		{
			printf("%d ", b[i][j]);
		}
		printf("\n");
	}
}

int main()
{
	reset();
	seq();
	print();
	reset();
	par();
	print();
}