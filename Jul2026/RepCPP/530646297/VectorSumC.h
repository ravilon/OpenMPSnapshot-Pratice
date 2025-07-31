#pragma once
using namespace std; //std::cout
#include <iostream> //cout, endl
#include <stdlib.h> //atoi, rand
#include <stdio.h> //printf
#include <omp.h> //omp


void initVector(int* v,int n_elems) {
	//cout << "vec_size: "<< vec_size << endl;
	//cout << "vector: ";
	for (unsigned i = 0; i < n_elems; i++) {
		//v[i] = rand() % 100;
		v[i] = rand() % 10;
		//cout << v[i] << " ";
	}
	//cout << "\n\n";
}


void printVector(int* v, int n_elems) {
	for (unsigned i = 0; i < n_elems; i++) {
		printf(" % d ", v[i]);
	}
	printf("\n\n");
}


void sumOfVectorsInC(int* a, int* b, int* c, int n_elems) {
	for (int i = 0; i < n_elems; i++) {
		c[i] = a[i] + b[i];
	}
}


void vectorSumC(int n_vector_elements, int trials) {
	
	const int N_ELEMS = n_vector_elements;
	size_t N_BYTES = N_ELEMS * sizeof(float);
	int* a;
	int* b;
	int* c;

	a = (int*)malloc(N_BYTES);
	b = (int*)malloc(N_BYTES);
	c = (int*)malloc(N_BYTES);

	initVector(a, N_ELEMS);
	initVector(b, N_ELEMS);

	double t1, t2;
	t1 = omp_get_wtime();
	for (int i = 0; i < trials; i++)
	{
		sumOfVectorsInC(a, b, c, N_ELEMS);
	}
	t2 = omp_get_wtime();
	printf("Tiempo medio en %d iteraciones de sumOfVectorsInC() con vector de %d elementos: % lf seconds.\n", trials, N_ELEMS, (t2 - t1) / (float)trials);

	//printVector(a, N_ELEMS);
	//printf("+\n\n");
	//printVector(b, N_ELEMS);
	//printf("=\n");
	//printVector(c, N_ELEMS);

	free(a);
	free(b);
	free(c);
}
