#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "hist-equ.h"


void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin){
    int i, threads_number;

    #pragma omp parallel for schedule(static)
    /* Este pragma paraleliza el bucle de inicialización del histograma `hist_out` a ceros. Usamos:
        - `schedule(static)`: Divide las iteraciones del bucle equitativamente entre los hilos de manera estática, útil porque todas las iteraciones tienen el mismo costo computacional.*/
    for ( i = 0; i < nbr_bin; i ++){
        hist_out[i] = 0;
    }

    #pragma omp parallel for reduction(+: hist_out[:nbr_bin])
    /*Este pragma paraleliza el bucle de construcción del histograma. Usamos:
    - `reduction(+: hist_out[:nbr_bin])`: Permite que cada hilo mantenga una copia privada del arreglo `hist_out`, acumulando resultados en paralelo sin interferencias. 
    Al finalizar, las copias privadas se combinan (reducción) en un único histograma final.*/
    for ( i = 0; i < img_size; i ++){
        hist_out[img_in[i]] ++;
    }
}

void histogram_equalization(unsigned char * img_out, unsigned char * img_in,
                            int * hist_in, int img_size, int nbr_bin){
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int i, cdf, min, d;
    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;
    
    #pragma omp parallel
    /* Se inicia una región paralela en la que cada hilo busca un valor mínimo no nulo localmente y luego actualiza una variable compartida `min` de forma segura usando una sección crítica.
        - `int min_local = 0;`: Cada hilo tiene su copia privada de `min_local` para calcular su propio mínimo sin interferencias.*/
    {
        int min_local = 0;

        #pragma omp parallel for simd
        /*Este pragma combina paralelismo en hilos con vectorización (SIMD) para recorrer el histograma (`hist_in`) y encontrar el valor mínimo no nulo en el bloque asignado a cada hilo.
         Cada hilo opera de forma independiente y vectorizada, evaluando los valores del histograma.*/
        for(i = 0; i < nbr_bin; i++){
            if (min_local == 0 && hist_in[i] != 0){
                min_local = hist_in[i];
            }
        }
        
        #pragma omp critical
       /* Sección crítica para actualizar la variable `min`. Esto evita condiciones de carrera al comparar y asignar el valor mínimo encontrado por los hilos.
            - `if (min == 0 || (min_local != 0 && min_local > min))`: Actualiza `min` solo si:
                - Es la primera vez que se encuentra un mínimo (`min == 0`), o
                - `min_local` es no nulo y mayor que el valor actual de `min`. */
        if (min == 0 || (min_local != 0 && min_local > min)){
            min = min_local;
        }
    }


    d = img_size - min;
    /*este caso se ha preferido no paralelizar debido a la variable cdf que se debe actualizar en cada iteracion
    se ha evaluado la creacion de un vector que guardase sus valores para su posterior uso con malloc y free,
    pero se considero que no merecia la pena*/
    for(i = 0; i < nbr_bin; i ++){
        cdf += hist_in[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if(lut[i] < 0){
            lut[i] = 0;
        }
    }
    

    /* Get the result image */
    #pragma omp parallel for schedule(static)
    /*Al igual que en casos anteriores, para la paralelizacion de este bucle se ha utilizado un schedule static para dividir de forma equitativa
    las ejecuciones del bucle para cada hilo*/
    for(i = 0; i < img_size; i ++){
        if(lut[img_in[i]] > 255){
            img_out[i] = 255;
        }
        else{
            img_out[i] = (unsigned char)lut[img_in[i]];
        }
    }
}



