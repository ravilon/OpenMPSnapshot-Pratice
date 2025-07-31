#pragma once
using namespace std; //std::cout
#include <iostream> //cout, endl
#include <stdlib.h> //atoi, rand
#include <stdio.h> //printf
#include <omp.h> //OpenMP


void showVector(const int* v, int n_elements) {
    for (int i = 0; i < n_elements; i++) {
        printf(" % d ", v[i]);
    }
    printf("\n\n");
}

void initializeVector(int* v, int n_elements) {
    //cout << "vec_size: "<< vec_size << endl;
    //cout << "vector: ";
    for (int i = 0; i < n_elements; i++) {
        //v[i] = rand() % 100;
        v[i] = rand() % 10;
        //cout << v[i] << " ";
    }
    //cout << "\n\n";
}



static void vectorSumOMP(int* a, int* b, int* c, int n_elements) {

    //omp_set_num_threads(nthreads);

    //La gestion de threads es automatica: Sin preguntar por id, 
    //nmero de threads, asignar carga a cada thread, etc.
    #pragma omp parallel for
    for (int i = 0; i < n_elements; i++) {
        c[i] = a[i] + b[i];
        //printf("Thread %d adding elements %d and %d in pos %d of the two vectors\n", omp_get_thread_num(), a[i], b[i], i);
    }
}

void vectorSumOpenMP(int n_elements, int trials, int n_threads) {

	const int N_ELEMENTS = n_elements;
	size_t N_BYTES = N_ELEMENTS * sizeof(int);

    //We reserve dynamic memory with malloc()
    //malloc() returns a void* pointer so we have to cast it to the desired pointer type
    int* a = (int*)malloc(N_BYTES);
    int* b = (int*)malloc(N_BYTES);
    int* c = (int*)malloc(N_BYTES);

    initializeVector(a, N_ELEMENTS);
    initializeVector(b, N_ELEMENTS);

    //Suma de vectores con OpenMP
    omp_set_num_threads(n_threads);
    double t1, t2;
    t1 = omp_get_wtime();
    for (int i = 0; i < trials; i++)
    {
        vectorSumOMP(a, b, c, N_ELEMENTS);

    }
    t2 = omp_get_wtime();
    printf("Tiempo medio en %d iteraciones de vectorSumOMP() con %d threads y con vector de %d elementos: % lf seconds.\n", trials, n_threads, N_ELEMENTS, (t2 - t1) / (float)trials);

    //showVector(a, N_ELEMENTS);
    //printf("+\n\n");
    //showVector(b, N_ELEMENTS);
    //printf("=\n\n");
    //showVector(c, N_ELEMENTS);

    free(a);
    free(b);
    free(c);
}
