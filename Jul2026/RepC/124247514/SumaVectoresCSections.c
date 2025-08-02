/* SumaVectoresCSections.c
Suma de dos vectores: v3 = v1 + v2
Para compilar usar (-lrt: real time library):
gcc -O2 SumaVectores.c -o SumaVectores lrt
gcc -O2 S SumaVectores.c lrt //para generar el cdigo ensamblador
Para ejecutar use: SumaVectoresC longitud
*/
#include <stdlib.h> // biblioteca con funciones atoi(), malloc() y free()
#include <stdio.h> // biblioteca donde se encuentra la funcin printf()
#include <time.h> // biblioteca donde se encuentra la funcin clock_gettime()
#include <omp.h> // biblioteca donde se encuentran las funciones OMP.
//#define PRINTF_ALL // comentar para quitar el printf ...
// que imprime todos los componentes
//Slo puede estar definida una de las tres constantes VECTOR_ (slo uno de los ...
//tres defines siguientes puede estar descomentado):
//#define VECTOR_LOCAL // descomentar para que los vectores sean variables ...
// locales (si se supera el tamao de la pila se ...
// generar el error "Violacin de Segmento")
#define VECTOR_GLOBAL// descomentar para que los vectores sean variables ...
// globales (su longitud no estar limitada por el ...
// tamao de la pila del programa)
//#define VECTOR_DYNAMIC // descomentar para que los vectores sean variables ...
// dinmicas (memoria reutilizable durante la ejecucin)

#ifdef VECTOR_GLOBAL
#define MAX 33554432 //=2^25
double v1[MAX], v2[MAX], v3[MAX];
#endif

int main(int argc, char** argv){

int i;
struct timespec cgt1,cgt2; double ncgt; //para tiempo de ejecucin
//Leer argumento de entrada (n de componentes del vector)
if (argc<2){
    printf("Faltan n componentes del vector\n");
    exit(-1);
}

unsigned int N = atoi(argv[1]); // Mximo N =2^32-1=4294967295 (sizeof(unsigned int) = 4 B)

#ifdef VECTOR_LOCAL
double v1[N], v2[N], v3[N]; // Tamao variable local en tiempo de ejecucin ...
// disponible en C a partir de actualizacin C99
#endif

#ifdef VECTOR_GLOBAL
if (N>MAX) N=MAX;
#endif

#ifdef VECTOR_DYNAMIC

double *v1, *v2, *v3;
v1 = (double*) malloc(N*sizeof(double));// malloc necesita el tamao en bytes
v2 = (double*) malloc(N*sizeof(double)); //si no hay espacio suficiente malloc devuelve NULL
v3 = (double*) malloc(N*sizeof(double));
if ( (v1==NULL) || (v2==NULL) || (v3==NULL) ){
    printf("Error en la reserva de espacio para los vectores\n");
    exit(-2);
}
#endif

//Inicializar vectores

#pragma omp parallel // Combinamos las directivas para usar solo 1 barrera implcita
{
    #pragma omp sections private (i)
    {
        #pragma omp section
            for(i=0; i<N/4; i++){
                v1[i] = N*0.1+i*0.1;
                v2[i] = N*0.1-i*0.1; //los valores dependen de N
                //printf("La hebra %d escribe V1[%d]=%f, V2[%d]=%f.\n", omp_get_thread_num(),i,v1[i],i,v2[i]);
            }

        #pragma omp section
            for(i=N/4; i < N/2; i++){
                v1[i] = N*0.1+i*0.1;
                v2[i] = N*0.1-i*0.1; //los valores dependen de N
                //printf("La hebra %d escribe V1[%d]=%f, V2[%d]=%f.\n", omp_get_thread_num(),i,v1[i],i,v2[i]);
            }

        #pragma omp section
            for(i=N/2; i < N*3/4; i++){
                v1[i] = N*0.1+i*0.1;
                v2[i] = N*0.1-i*0.1; //los valores dependen de N
                //printf("La hebra %d escribe V1[%d]=%f, V2[%d]=%f.\n", omp_get_thread_num(),i,v1[i],i,v2[i]);
            }

        #pragma omp section
            for(i=N*3/4; i < N; i++){
                v1[i] = N*0.1+i*0.1;
                v2[i] = N*0.1-i*0.1; //los valores dependen de N
                //printf("La hebra %d escribe V1[%d]=%f, V2[%d]=%f.\n", omp_get_thread_num(),i,v1[i],i,v2[i]);
            }

    }
}

double start = omp_get_wtime();
//Calcular suma de vectores
#pragma omp parallel
{
    #pragma omp sections private(i)
    {
        #pragma omp section
            for(i=0; i<N/4; i++)
                v3[i] = v1[i] + v2[i];


        #pragma omp section
            for(i=N/4; i<N/2; i++)
                v3[i] = v1[i] + v2[i];


        #pragma omp section
            for(i=N/2; i<N*3/4; i++)
                v3[i] = v1[i] + v2[i];


        #pragma omp section
            for(i=N*3/4; i<N; i++)
                v3[i] = v1[i] + v2[i];

    }

}


double end = omp_get_wtime();
double diff = end - start;
//Imprimir resultado de la suma y el tiempo de ejecucin
#ifdef PRINTF_ALL
printf("Tiempo(seg.):%11.9f\t / Tamao Vectores:%u\n",diff,N);
for(i=0; i<N; i++)
printf("/ V1[%d]+V2[%d]=V3[%d](%8.6f+%8.6f=%8.6f) /\n",
i,i,i,v1[i],v2[i],v3[i]);
#else
/*
printf("Tiempo(seg.):%11.9f\t / Tamao Vectores:%u\t/ V1[0]+V2[0]=V3[0](%8.6f+%8.6f=%8.6f) / \
       V1[%d]+V2[%d]=V3[%d](%8.6f+%8.6f=%8.6f) /\n",
diff,N,v1[0],v2[0],v3[0],N-1,N-1,N-1,v1[N-1],v2[N-1],v3[N-1]);

if (N<=11)
    for(i=0; i<N; i++)
        printf("v1[%d]=%f, v2[%d]=%f, v3[%d]=%f\n",i,v1[i],i,v2[i],i,v3[i]);

printf("\n");
*/

printf("%f\n",diff);
#endif
#ifdef VECTOR_DYNAMIC
free(v1); // libera el espacio reservado para v1
free(v2); // libera el espacio reservado para v2
free(v3); // libera el espacio reservado para v3
#endif
return 0;
}
