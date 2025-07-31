#include <stdio.h>
#include <omp.h>

int main(int argc, char** argv) 
{
    int n = 1;
    // int N = 121;    // 121 120 60 30 15 14 7 6 3 2 1 -> 11
    // int N = 51;       // 51 50 25 24 12 6 3 2 1 -> 9
    // int N = 30;     // 30 15 14 7 6 3 2 1 -> 8
    // int N = 17;     // 17 16 8 4 2 1 -> 6

    long indice = 0;
    long antecesor = 0, sucesor = 0;

    printf("Ingrese valor de n: ");
    scanf("%d", &n);

    antecesor = n;
    
    // Usar OpenMP para paralelizar el bucle while
    #pragma omp parallel reduction(+:indice) // private(antecesor, sucesor)
    {
        //while (antecesor != 0) 
        for(; antecesor != 0;)
        {
            // Cada hilo procesa una iteración del bucle
            if (antecesor % 2 == 0 && antecesor / 2 > 1)
                sucesor = antecesor / 2;
            else
                sucesor = antecesor - 1;

            antecesor = sucesor;
            indice++;
        }
    }

    printf("%ld\n", indice);

    return 0;
}
