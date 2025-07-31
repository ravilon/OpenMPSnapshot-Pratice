/*
 ============================================================================
 Name        : Lab7_p1_openmp.c
 Author      : Alexandru Grigoras
 Version     : 0.1
 Copyright   : Alexandru Grigoras
 Description : Muller-Preparata sorting algorithm
 ============================================================================
 */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

void Interschimba(int &a, int &b)
{
	int aux = a;
	a = b;
	b = aux;
}

void AfisareVector(int v[], int n)
{
	int i;

	for (i = 0; i < n; i++)
	{
		printf("%d ", v[i]);
	}

	printf("\n");
}

void AfisareMatrice(int **a, int n, int m)
{
	int i, j;
	for (i = 0; i < n; ++i)
	{
		for (j = 0; j < m; ++j)
		{
			printf("%2d ", a[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

void InitializareMatrice(int **v, int n, int m)
{
	int i, j;

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < m; j++)
		{
			v[i][j] = 0;
		}
	}
}

// functia de comprimare (reducere), algoritmul paralel
void ComprimParalela(int **R, int j, int n)
{
	int m = 2;

	// comprimarea se face pe coloana j
	int k, i;
	int doi_la_k;
	int doi_la_k_plus_1;

	for (k = m - 1; k >= 0; --k)
	{
		doi_la_k = 1 << k;
		doi_la_k_plus_1 = 1 << (k + 1);
//#pragma omp parallel for shared(R)
		for (i = doi_la_k; i <= doi_la_k_plus_1 - 1; ++i)
		{
			// se stocheaza pe indicele liniei i, decalat cu -1 (de aceea R[i - 1][j])
			// deoarece coloanei j ii lipseste un element pana la un numar de elemente putere a lui 2
			// vectorii coloana sunt "shiftati" in sus cu un element
			R[i - 1][j] = R[2 * i - 1][j] + R[2 * i][j];
		}
	}
}

void SortareParalelaMullerPreparata(int *A, int n)
{
	int i, j, tid;
	int *P;
	int *B;
	int **R;

	P = (int*) malloc(n*sizeof(int));
	B = (int*) malloc(n * sizeof(int));
	R = (int**) malloc((2 * n - 1) * sizeof(int));
	for (i = 0; i < 2 * n - 1; ++i)
	{
		R[i] = (int*)malloc(n * sizeof(int));
		for (j = 0; j < n; j++)
		{
			R[i][j] = 0;
		}
	}

	InitializareMatrice(R, 2 * n - 1, n);

#pragma omp parallel for private(i)
	for (j = 0; j <= n - 1; ++j)
	{
		for (i = 0; i <= n - 1; ++i)
		{
			if (A[i] < A[j])
			{
				R[i + n - 1][j] = 1;
				//printf("R[%d][%d] = %d\n", i + n - 1, j, R[i + n - 1][j]);
			}
			else
			{
				R[i + n - 1][j] = 0;
				//printf("R[%d][%d] = %d\n", i + n - 1, j, R[i + n - 1][j]);
			}
			//tid = omp_get_thread_num();
			//printf("thread %d\n", tid);
		}
		printf("\n");
	}
	
	AfisareMatrice(R, 2 * n - 1, n);

#pragma omp parallel for shared(R, P, j, n)
	for (j = 0; j <= n - 1; j++)
	{
		ComprimParalela(R, j, n);
		P[j] = R[0][j];
	}

	AfisareMatrice(R, 2 * n - 1, n);

#pragma omp parallel for
	for (j = 0; j <= n - 1; j++)
	{
		// Solutia 1: neeficienta
		//B[P[j]] = A[j];

		// Solutia 2: eficienta
		if (j > P[j])
		{
			Interschimba(A[P[j]], A[j]);
			Interschimba(P[j], P[P[j]]);
		}
	}

	AfisareVector(A, n);

	if (B)
	{
		free(B);
	}
	if (P)
	{
		free(P);
	}
	if (R)
	{
		for (i = 0; i < 2 * n - 1; ++i)
		{
			if (R[i])
			{
				free(R[i]);
			}
		}
		free(R);
	}
}

int main(int argc, char *argv[]) {

	int tid;
	int nrElem = 4;
	int A[4] = { 2, 6, 3, 8 };

	omp_set_num_threads(nrElem);

	AfisareVector(A, nrElem);

	SortareParalelaMullerPreparata(A, nrElem);

	//printf("sortata: ");
	//AfisareVector(A, nrElem);

	// eliberam memoria
	if (A)
	{
		free(A);
	}

	return 0;
}


