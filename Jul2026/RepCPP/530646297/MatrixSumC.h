#pragma once
using namespace std; //std::cout
#include <iostream> //cout, endl
#include <stdlib.h> //atoi, rand
#include <stdio.h> //printf
#include <omp.h> //omp



void initMatrixVector(int* v, const unsigned vec_size) {
	//cout << "vec_size: "<< vec_size << endl;
	//cout << "vector: ";
	for (unsigned i = 0; i < vec_size; i++) {
		//v[i] = rand() % 100;
		v[i] = rand() % 100;
		//cout << v[i] << " ";
	}
	//cout << "\n\n";
}


void printMatrix(const int* m, const unsigned matrixWidth, const unsigned matrixHeight) {
	printf("\n");
	for (unsigned i = 0; i < matrixHeight; i++) {
		for (unsigned j = 0; j < matrixWidth; j++) {
			printf("%d ", m[j + i * matrixWidth]);
		}
		printf("\n");
	}
	printf("\n");
}


static void matrixSum1for(int* a, int* b, int* c, const unsigned matrixWidth, const unsigned matrixHeight) {
	for (unsigned i = 0; i < matrixHeight * matrixWidth; i++) {
		c[i] = a[i] + b[i];
	}
}

static void matrixSum2for(int* a, int* b, int* c, const unsigned matrixWidth, const unsigned matrixHeight) {
	for (unsigned i = 0; i < matrixHeight; i++) {
		for (int j = 0; j < matrixWidth; j++) {
			c[j + i * matrixWidth] = a[j + i * matrixWidth] + b[j + i * matrixWidth];
		}
	}
}


void matrixSumCOneLoop(int width, int height, int trials) {
	printf("MatrixSumC 1 loop");

	const int MATRIX_WIDTH = width;
	const int MATRIX_HEIGHT = height;
	const int N_ELEMENTS = MATRIX_WIDTH * MATRIX_HEIGHT;

	//each matrix is created as a vector of size MxN
	int* a = (int*)malloc(N_ELEMENTS * sizeof(int));
	int* b = (int*)malloc(N_ELEMENTS * sizeof(int));
	int* c = (int*)malloc(N_ELEMENTS * sizeof(int));

	initMatrixVector(a, N_ELEMENTS);
	initMatrixVector(b, N_ELEMENTS);

	double t1, t2;
	t1 = omp_get_wtime();
	for (int i = 0; i < trials; i++)
	{
		matrixSum1for(a, b, c, MATRIX_WIDTH, MATRIX_HEIGHT);
	}
	t2 = omp_get_wtime();
	printf("Tiempo medio en %d iteraciones de matrixSumCOneLoop() con matriz de %dx%d elementos: % lf seconds.\n", trials, MATRIX_WIDTH, MATRIX_HEIGHT, (t2 - t1) / (float)trials);

	//showMatrix(a, MATRIX_WIDTH, MATRIX_HEIGHT);
	//printf("+\n");
	//showMatrix(b, MATRIX_WIDTH, MATRIX_HEIGHT);
	//printf("=\n");
	//showMatrix(c, MATRIX_WIDTH, MATRIX_HEIGHT);

	free(a);
	free(b);
	free(c);
}

void matrixSumCTwoLoops(int width, int height, int trials) {
	printf("MatrixSumC 2 loops");

	const int MATRIX_WIDTH = width;
	const int MATRIX_HEIGHT = height;
	const int N_ELEMENTS = MATRIX_WIDTH * MATRIX_HEIGHT;

	//each matrix is created as a vector of size MxN
	int* a = (int*)malloc(N_ELEMENTS * sizeof(int));
	int* b = (int*)malloc(N_ELEMENTS * sizeof(int));
	int* c = (int*)malloc(N_ELEMENTS * sizeof(int));

	initMatrixVector(a, N_ELEMENTS);
	initMatrixVector(b, N_ELEMENTS);

	double t1, t2;
	t1 = omp_get_wtime();
	for (int i = 0; i < trials; i++)
	{
		matrixSum2for(a, b, c, MATRIX_WIDTH, MATRIX_HEIGHT);
	}
	t2 = omp_get_wtime();
	printf("Tiempo medio en %d iteraciones de matrixSumCTwoLoops() con matriz de %dx%d elementos: % lf seconds.\n", trials, MATRIX_WIDTH, MATRIX_HEIGHT, (t2 - t1) / (float)trials);

	//printMatrix(a, MATRIX_WIDTH, MATRIX_HEIGHT);
	//printf("+\n");
	//printMatrix(b, MATRIX_WIDTH, MATRIX_HEIGHT);
	//printf("=\n");
	//printMatrix(c, MATRIX_WIDTH, MATRIX_HEIGHT);

	free(a);
	free(b);
	free(c);
}

