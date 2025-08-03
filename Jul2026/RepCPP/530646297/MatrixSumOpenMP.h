#pragma once
using namespace std; //std::cout
#include <iostream> //cout, endl
#include <stdlib.h> //atoi, rand
#include <stdio.h> //printf
#include <omp.h> //OpenMP


void showMatrix(const int* v, int n_elements) {
    for (unsigned i = 0; i < n_elements; i++) {
        printf(" % d ", v[i]);
    }
    printf("\n\n");
}

void initializeMatrixVector(int* v, int n_elements) {
    //cout << "vec_size: "<< vec_size << endl;
    //cout << "vector: ";
    for (unsigned i = 0; i < n_elements; i++) {
        //v[i] = rand() % 100;
        v[i] = rand() % 10;
        //cout << v[i] << " ";
    }
    //cout << "\n\n";
}





static void matrixSumOMP1for(int* a, int* b, int* c, int n_elements, int nthreads) {


    //La gestion de threads es automatica: Sin preguntar por id, 
    //nmero de threads, asignar carga a cada thread, etc.
    #pragma omp parallel for
    for (int i = 0; i < n_elements; i++) {
        c[i] = a[i] + b[i];
        //printf("Thread %d adding elements %d and %d in pos %d of the two vectors\n", omp_get_thread_num(), a[i], b[i], i);
    }
}


static void matrixSumOMP2forAndInternalLoopParallelized(int* a, int* b, int* c, int matrixWidth, int matrixHeight, int nthreads) {

    #pragma omp parallel
    {
        for (int i = 0; i < matrixHeight; i++) {
            #pragma omp for //each thread sums one element of each row of matrix a and b
            for (int j = 0; j < matrixWidth; j++) {
                c[j + i * matrixWidth] = a[j + i * matrixWidth] + b[j + i * matrixWidth];
            }
        }
    }
}

static void matrixSumOMP2forAndExternalLoopParallelized(int* a, int* b, int* c, int matrixWidth, int matrixHeight, int nthreads) {

    #pragma omp parallel
    {
        #pragma omp for //each thread sums one row of each matrix a and b
        for (int i = 0; i < matrixHeight; i++) {
            for (int j = 0; j < matrixWidth; j++) {
                c[j + i * matrixWidth] = a[j + i * matrixWidth] + b[j + i * matrixWidth];
            }
        }
    }
}

static void matrixSumOMP2forAndBothLoopsParallelized(int* a, int* b, int* c, const unsigned matrixWidth, const unsigned matrixHeight, const unsigned nthreads) {

    omp_set_num_threads(nthreads);

    #pragma omp parallel
    {
        //Allows parallelization of perfectly nested loops without using nested parallelism. 
        //Compiler forms a single loop and then parallelizes this.
        //Keep in mind that some loops are not collapsable due to data dependency (but not in this case).
        #pragma omp for collapse(2)
        for (int i = 0; i < matrixHeight; i++) {
            for (int j = 0; j < matrixWidth; j++) {
                c[j + i * matrixWidth] = a[j + i * matrixWidth] + b[j + i * matrixWidth];
            }
        }
    }
}


//1 for
void matrixSumOpenMPOneLoop(int width, int height, int trials, int n_threads) {
    printf("matrixSumOpenMP OneLoop");

    const int WIDTH = width;
    const int HEIGHT = height;
    const int N_ELEMENTS = width * height;
    size_t N_BYTES = N_ELEMENTS * sizeof(int);


    //We reserve dynamic memory with malloc()
    //malloc() returns a void* pointer so we have to cast it to the desired pointer type
    int* a = (int*)malloc(N_BYTES * sizeof(int));
    int* b = (int*)malloc(N_BYTES * sizeof(int));
    int* c = (int*)malloc(N_BYTES * sizeof(int));

    initializeMatrixVector(a, N_ELEMENTS);
    initializeMatrixVector(b, N_ELEMENTS);

    omp_set_num_threads(n_threads);
    double t1, t2;
    t1 = omp_get_wtime();
    for (int i = 0; i < trials; i++)
    {
        matrixSumOMP1for(a, b, c, N_ELEMENTS, n_threads);

    }
    t2 = omp_get_wtime();
    printf("Tiempo medio en %d iteraciones de matrixSumOMP1for() con %d threads y con matriz de %dx%d elementos: % lf seconds.\n", trials, n_threads, WIDTH,HEIGHT, (t2 - t1) / (float)trials);

    //showMatrix(a, N_ELEMENTS);
    //printf("+\n\n");
    //showMatrix(b, N_ELEMENTS);
    //printf("=\n\n");
    //showMatrix(c, N_ELEMENTS);

    free(a);
    free(b);
    free(c);
}


//external
void matrixSumOpenMPTwoLoopsAndInternalLoopParallelized(int width, int height, int trials, int n_threads) {
    printf("matrixSumOpenMP TwoLoops And InternalLoopParallelized");

    const int WIDTH = width;
    const int HEIGHT = height;
    const int N_ELEMENTS = width * height;
    size_t N_BYTES = N_ELEMENTS * sizeof(int);


    //We reserve dynamic memory with malloc()
    //malloc() returns a void* pointer so we have to cast it to the desired pointer type
    int* a = (int*)malloc(N_BYTES);
    int* b = (int*)malloc(N_BYTES);
    int* c = (int*)malloc(N_BYTES);

    initializeMatrixVector(a, N_ELEMENTS);
    initializeMatrixVector(b, N_ELEMENTS);

    omp_set_num_threads(n_threads);
    double t1, t2;
    t1 = omp_get_wtime();
    for (int i = 0; i < trials; i++)
    {
        matrixSumOMP2forAndInternalLoopParallelized(a, b, c, WIDTH,HEIGHT, n_threads);

    }
    t2 = omp_get_wtime();
    printf("Tiempo medio en %d iteraciones de matrixSumOMP2forAndInternalLoopParallelized() con %d threads y con matriz de %dx%d elementos: % lf seconds.\n", trials, n_threads, WIDTH, HEIGHT, (t2 - t1) / (float)trials);

    //showMatrix(a, N_ELEMENTS);
    //printf("+\n\n");
    //showMatrix(b, N_ELEMENTS);
    //printf("=\n\n");
    //showMatrix(c, N_ELEMENTS);

    free(a);
    free(b);
    free(c);
}


//internal
void matrixSumOpenMPTwoLoopsAndExternalLoopParallelized(int width, int height, int trials, int n_threads) {
    printf("matrixSumOpenMP TwoLoops And ExternalLoopParallelized");

    const int WIDTH = width;
    const int HEIGHT = height;
    const int N_ELEMENTS = width * height;
    size_t N_BYTES = N_ELEMENTS * sizeof(int);


    //We reserve dynamic memory with malloc()
    //malloc() returns a void* pointer so we have to cast it to the desired pointer type
    int* a = (int*)malloc(N_BYTES);
    int* b = (int*)malloc(N_BYTES);
    int* c = (int*)malloc(N_BYTES);

    initializeMatrixVector(a, N_ELEMENTS);
    initializeMatrixVector(b, N_ELEMENTS);

    omp_set_num_threads(n_threads);
    double t1, t2;
    t1 = omp_get_wtime();
    for (int i = 0; i < trials; i++)
    {
        matrixSumOMP2forAndExternalLoopParallelized(a, b, c, WIDTH, HEIGHT, n_threads);

    }
    t2 = omp_get_wtime();
    printf("Tiempo medio en %d iteraciones de matrixSumOMP2forAndExternalLoopParallelized() con %d threads y con matriz de %dx%d elementos: % lf seconds.\n", trials, n_threads, WIDTH, HEIGHT, (t2 - t1) / (float)trials);

    //showMatrix(a, N_ELEMENTS);
    //printf("+\n\n");
    //showMatrix(b, N_ELEMENTS);
    //printf("=\n\n");
    //showMatrix(c, N_ELEMENTS);

    free(a);
    free(b);
    free(c);
}


//both
void matrixSumOpenMPTwoLoopsAndBothLoopsParallelized(int width, int height, int trials, int n_threads) {
    printf("matrixSumOpenMP TwoLoops And BothLoopsParallelized");

    const int WIDTH = width;
    const int HEIGHT = height;
    const int N_ELEMENTS = width * height;
    size_t N_BYTES = N_ELEMENTS * sizeof(int);


    //We reserve dynamic memory with malloc()
    //malloc() returns a void* pointer so we have to cast it to the desired pointer type
    int* a = (int*)malloc(N_BYTES);
    int* b = (int*)malloc(N_BYTES);
    int* c = (int*)malloc(N_BYTES);

    initializeMatrixVector(a, N_ELEMENTS);
    initializeMatrixVector(b, N_ELEMENTS);

    omp_set_num_threads(n_threads);
    double t1, t2;
    t1 = omp_get_wtime();
    for (int i = 0; i < trials; i++)
    {
        matrixSumOMP2forAndBothLoopsParallelized(a, b, c, WIDTH, HEIGHT, n_threads);

    }
    t2 = omp_get_wtime();
    printf("Tiempo medio en %d iteraciones de matrixSumOMP2forAndBothLoopsParallelized() con %d threads y con matriz de %dx%d elementos: % lf seconds.\n", trials, n_threads, WIDTH, HEIGHT, (t2 - t1) / (float)trials);

    //showMatrix(a, N_ELEMENTS);
    //printf("+\n\n");
    //showMatrix(b, N_ELEMENTS);
    //printf("=\n\n");
    //showMatrix(c, N_ELEMENTS);

    free(a);
    free(b);
    free(c);
}

