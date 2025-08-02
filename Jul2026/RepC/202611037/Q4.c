/*
Compilar: gcc -o Q4 Q4.c -ansi -Wall -fopenmp
Executar: ./Q4 4
*/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 1022 /*N é a dimensão da placa quadrada*/
#define M 1024 /*M número de iterações*/
#define TAM N+2 
double temp[TAM][TAM], mat[TAM][TAM];

void sor(void);

int main(int argc, char * argv[]){
    int i, j;
    double t_final, t_inicial;

    if(argc < 2){
        perror("Número de argumentos insuficiente, insira a quantidade de threads.");
        return 0;
    }else{
        omp_set_num_threads(atoi(argv[1]));
    }


    /*Temperatura inicial da placa é de 20°C*/
    for(i=0; i<TAM; i++){
        for(j=0; j<TAM; j++){
            mat[i][j] = 20;
        }
    }

    /*Tempratura da fonte de calor localizada no ponto 800x800 da placa é de 100°C*/
    mat[800][800] = 100;

    t_inicial = omp_get_wtime();
    sor();
    t_final = omp_get_wtime();


    printf("Tempo de execução: %lf\n", t_final-t_inicial);

    return 0;
}

void sor(void){
    int i, j, k;
    #pragma omp parallel shared(mat, temp) private(i, j)
    {
        #pragma omp single
        {
            for(k=0; k < M; k++){
                #pragma omp taskgroup
                {
                    #pragma omp taskloop
                    for(i=1; i<TAM-1; i++){
                        for(j=1; j<TAM-1; j++){
                            temp[i][j] = 0.25*(mat[i-1][j] + mat[i+1][j] + mat[i][j-1] + mat[i][j+1]);
                        }
                    }
                }
                //#pragma omp taskwait
                memcpy(mat, temp, sizeof(mat));
            }
            //#pragma omp barrier
        }
    }
}