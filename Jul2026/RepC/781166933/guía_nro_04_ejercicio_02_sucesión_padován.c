#include <stdio.h>
#include <omp.h>

#define N 11

int main()
{   
    int sucesion[N], *puntero, i, n_2, n_3, total = 1 ;

    sucesion[0] = 1;
    sucesion[1] = 0;
    sucesion[2] = 0;

    puntero = &sucesion[2];

    #pragma omp parallel for private(n_2, n_3) reduction(+:total)
    for( i = 3 ; i < N ; i++ )
    {  
        #pragma omp critical(zona_01)
        {
            puntero-=1;
            n_2 = *puntero;
            puntero-=1;
            n_3 = *puntero;
            puntero+=3;
            *puntero = n_2+n_3;
        }

        total += *puntero;
    }

    printf("Resultado es: %d\n",total);
}
