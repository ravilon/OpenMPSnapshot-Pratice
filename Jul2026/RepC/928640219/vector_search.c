#include <omp.h>
#include <stdio.h>
#include <conio.h>
#include <stdlib.h>
#include <time.h>

int find_max(int v[])
{
    int max = v[0];
    #pragma omp parallel
    {
        #pragma omp for reduction(max:max)
        for (int i=0; i<100; i++)
        {
            if (v[i] > max)
                max = v[i]; 
        }
    }

    return max;
}

int find_max_sequencial(int v[]) {
    int max = v[0];
    for (int i = 0; i < 100; i++) {
        if (v[i] > max)
            max = v[i];
    }
    return max;
}

int find_number_parallel(int v[], int n)
{
    int index = -1;
    #pragma omp parallel for shared(index)
    for (int i = 0; i < 100; i++) {
        if (v[i] == n)
        {
            #pragma omp critical
            {
                index = i;
            }
        }
    }

    return index;
}

int main(){
    srand(time(0)); 
    omp_set_num_threads(4);

    int v[100];
    for (int i = 0; i<100; i++)
    {
        v[i] = rand() % 100;
    }

    printf("Max Paralelo: %d\n", find_max(v));
    printf("Max Sequencial: %d\n", find_max_sequencial(v));

    int index = find_number_parallel(v,1);
    printf("Numero encontrado na posicao %d. v[%d] = %d", index, index, v[index]);
}
