#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h> 
#define N 8
#define NUM_THREADS 8

void print_matrix(float matrix[N][N], int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)  
            printf("%f ", matrix[i][j]);
        printf("\n");
    }
}

void gaussian_paralel_v1(float a[N][N], int size)
{
    float l[N];
    omp_set_num_threads(NUM_THREADS);
    #pragma omp parallel shared(a, l)
    {
        for (int i = 0; i < size; i++)
        {
            #pragma omp for
            for (int j = i; j < size; j++)  
            {
                if (i != j) 
                    l[j] = a[j][i] / a[i][i];

                for (int k = 0; k < size; k++)
                { 
                    if (j == i) 
                        a[j][k] = a[i][k];
                    else 
                    {
                        if (k < j) 
                            a[j][k] = 0;
                        else 
                            a[j][k] = a[j][k] - l[j] * a[i][k];
                    }       
                }
            }
        }
    }
}

void gaussian_paralel_v2(float a[N][N], int size)
{
    float l[N];
    for (int i = 0; i < size; i++)
    {
        #pragma omp parallel for shared(a, l, i, size) 
        for (int j = i + 1; j < size; j++)  
        {
            if (a[i][i] != 0) {
                l[j] = a[j][i] / a[i][i];
            }
        }

        #pragma omp parallel for shared(a, l, i, size) 
        for (int j = i + 1; j < size; j++)  
        {
            for (int k = i; k < size; k++)  
            {
                a[j][k] = a[j][k] - l[j] * a[i][k];
            }
        }
    }
}

void gaussian(float a[N][N], int size)
{
    float l[N];
    #pragma omp parallel for private(l) shared(a, size) collapse(2)
    for (int i = 0; i < size; i++)
    {
        for (int j = i; j < size; j++)
        {
            if (i != j) {
                l[j] = a[j][i] / a[i][i];
            }
            for (int k = 0; k < size; k++)
            { 
                if (j == i) {
                    a[j][k] = a[i][k];
                } else {
                    if (k < j) {
                        a[j][k] = 0;
                    } else {
                        a[j][k] = a[j][k] - l[j] * a[i][k];
                    }
                }
            }    
        }
    }
}

void random_fill(float matrix[N][N], int size)
{
    srand(time(0));
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            matrix[i][j] = ((rand() % 20) + 1);
        }
    }
}

void copy_matrix(float src[N][N], float dest[N][N], int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            dest[i][j] = src[i][j];
        }
    }
}

int main(void)
{
    omp_set_num_threads(NUM_THREADS);
    int size = N;
    float a[N][N], b[N][N];
    random_fill(a, size);     
    copy_matrix(a, b, size);  

    //le code séquentiel 
    printf("*** Gaussian Séquentiel ***\n");
    clock_t start_t = clock();
    gaussian(a, size);
    clock_t end_t = clock();
    double running_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
    printf("***U Séquentiel***\n");
    print_matrix(a, size);
    printf("Le temps séquentiel pour la décomposition gaussienne = %f s.\n", running_t);

    //1ere version du code parallele 
    // double start_t = omp_get_wtime();
    // gaussian_paralel_v1(b, size);
    // double end_t = omp_get_wtime();
    // double running_t = end_t - start_t;
    // printf("Le temps parallèle pour la décomposition gaussienne v1 = %f s.\n", running_t);

    //2eme version 
    double start = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        gaussian_paralel_v2(a, size);
    }
    double end = omp_get_wtime();
    print_matrix(b, size);
    double running = end - start;
    printf("Le temps parallèle pour la décomposition gaussienne v2 = %f s.\n", running);

    //3eme version 
    double start_t = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < size; i++)
        {
            gaussian(a, size);
        }
    }

    double end_t = omp_get_wtime(); 
    double runing_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
    printf("\nLe temps parallel pour le calcul de la décomposition Gaussienne de la matrice = %f s.\n ", runing_t);
    return 0;
}

