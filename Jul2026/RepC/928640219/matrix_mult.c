#include <omp.h>
#include <stdio.h>
#include <conio.h>
#include <stdlib.h>
#include <time.h>

void matrix_mult(int a[3][3], int b[3][3],int c[3][3])
{
    int linhas = 3;
    int colunas = 3;

    #pragma omp parallel 
    {
        #pragma omp for
        for (int i=0; i<linhas; i++)
        {
            for (int j=0; j<colunas; j++)
            {
                c[i][j] = 0;
                for (int k=0; k<colunas; k++)
                {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    }
    
}

void matrix_mult_iterativa(int a[3][3], int b[3][3],int c[3][3])
{
    int linhas = 3;
    int colunas = 3;

    for (int i=0; i<linhas; i++)
    {
        for (int j=0; j<colunas; j++)
        {
            c[i][j] = 0;

            for (int k=0; k<colunas; k++)
            {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

int main(){
    srand(time(0)); 
    omp_set_num_threads(4);

    int m1[3][3];
    int m2[3][3];
    int mf1[3][3];
    int mf2[3][3];

    #pragma omp parallel for
    for (int i = 0; i<3; i++)
    {
        for (int j = 0; j<3; j++)
        {
            m1[i][j] = rand() % 10;
            m2[i][j] = rand() % 10;
            mf1[i][j] = 0;
            mf2[i][j] = 0;
        }
    }

    matrix_mult_iterativa(m1,m2,mf1);

    printf("Matriz Iterativa:\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%d ", mf1[i][j]);
        }
        printf("\n");
    }

    matrix_mult(m1,m2,mf2);
    printf("Matriz Paralela:\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%d ", mf2[i][j]);
        }
        printf("\n");
    }

    return 0;
}