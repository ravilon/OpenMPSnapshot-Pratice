#include <omp.h>
#include <stdio.h>

int somar(int ar[], int inicio, int fim)
{
    printf("Somando de %d a %d\n", inicio, fim-1);
    int soma = 0;
    for (int i=inicio; i<fim; i++)
    {
        soma = soma + ar[i];
    }
    return soma;
}

int main(){
    omp_set_num_threads(4);

    int v[100];
    for (int i=1; i<=100; i++)
        v[i-1] = i;

    int divisao = 100/3;

    int s1 = 0;
    int s2 = 0;
    int s3 = 0;

    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            s1 = somar(v, 0, divisao);

            #pragma omp task
            s2 = somar(v, divisao, divisao*2);

            #pragma omp task
            s3 = somar(v, divisao*2, divisao*3+1);

            #pragma omp taskwait
            printf("Todas as tarefas terminaram\n");

            printf("Soma: %d", s1+s2+s3);
        }
    }

}