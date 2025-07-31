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

int main(){
    srand(time(0)); 
    omp_set_num_threads(4);

    int v[100];
    for (int i = 0; i<100; i++)
    {
        v[i] = rand() % 200;
    }

    printf("Max Paralelo: %d\n", find_max(v));
    printf("Max Sequencial: %d", find_max_sequencial(v));
}
