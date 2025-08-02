#include <stdlib.h> // biblioteca con funciones atoi(), malloc() y free()
#include <stdio.h>  // biblioteca donde se encuentra la función printf()
#include <time.h> // biblioteca donde se encuentra la función clock_gettime() 
#include <omp.h>

//Sólo puede estar definida una de las tres constantes VECTOR_ (sólo uno de los ... 
//tres defines siguientes puede estar descomentado):
// #define VECTOR_GLOBAL // descomentar para que los vectores sean variables ...
      // globales (su longitud no estará limitada por el ...
      // tamaño de la pila del programa)
#define VECTOR_DYNAMIC  // descomentar para que los vectores sean variables ...
      // dinámicas (memoria reutilizable durante la ejecución)
    
#ifdef VECTOR_GLOBAL
    #define MAX 16383

    double v1[MAX], v2[MAX], m[MAX][MAX]; 
#endif

int main(int argc, char **argv)  {
    int i, j;

    if(argc < 2){
        printf("Faltan nº componentes del vector\n");
      exit(-1);
    }

    // Tamaño de la matriz pasado por parámetro
    unsigned int n = atoi(argv[1]);

    if(n < 2){
    	fprintf(stderr, "El tamaño debe ser mayor de 2");
    	exit(-1);
    }

    #ifdef VECTOR_DYNAMIC
        // Reserva de memoria para la versión dinámica
        double *v1, *v2, **m;

        v1 = (double*) malloc(n*sizeof(double));
        v2 = (double*) malloc(n*sizeof(double));
        m = (double**) malloc(n*sizeof(double*));

        for(i=0; i<n; i++){
            m[i] = (double*) malloc(n*sizeof(double));
        }
    #endif

    double t1, t2;

    #pragma omp parallel private(i)
    {
        // Inicialización --------------------
        for(i=0; i<n; i++){
            v1[i] = 3;
            v2[i] = 0;
        }

        for(i=0; i<n; i++){
            // #pragma omp for
            for(j=0; j<n; j++){
                m[i][j] = 4;
            }
        }

        // Calculo de la fórmula
        #pragma omp single
            t1 = omp_get_wtime();

        for(i=0; i<n; i++){
            double num = 0;

            #pragma omp for
            for(j=0; j<n; j++){
                num = num + (m[i][j]*v1[j]);
            }

            #pragma omp critical
            v2[i] += num;
        }

        #pragma omp single
            t2 = omp_get_wtime() - t1;
    }

    // Imprimo resultados
    if(n <= 15){
    	printf("Tiempo: %f\n Tamaño: %i\n", t2, n);

    	for(i=0; i<n; i++){
    		printf("v[%i] = %f\n", i, v2[i]);
    	}
    }
    else{
    	printf("Tiempo: %f\n Tamaño: %i\n Primera componente: %f\n Última componente: %f\n", t2, n, v2[0], v2[n-1]);
    }

    #ifdef VECTOR_DYNAMIC
        // Liberamos memoria
        free(v1);
        free(v2);

        for(i=0; i<n; i++){
            free(m[i]);
        }

        free(m);
    #endif

    return 0;
}