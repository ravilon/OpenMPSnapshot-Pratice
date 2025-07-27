/*Para compilar usar (-lrt: real time library): gcc -O2 Sumavectores.c -o SumaVectores -lrt
Para ejecutar use: SumaVectores longitud
*/

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <time.h>

#define PRINTF_ALL
//Sólo puede estar definida una de las tres constantes VECTOR_ (sólo uno de los ...
//tres defines siguientes puede estar descomentado):
//#define VECTOR_LOCAL
  // descomentar para que los vectores sean variables ...
  // locales (si se supera el tamaño de la pila se ...
  // generará el error "Violación de Segmento")
//#define VECTOR_GLOBAL // descomentar para que los vectores sean variables ...
  // globales (su longitud no estará limitada por el ...
  // tamaño de la pila del programa)

#define VECTOR_DYNAMIC //descomentar para que los vectores sean variables ...
		      //dinámicas (memoria reautilizable durante la ejecución)

#ifdef VECTOR_GLOBAL
#define MAX 33554432
double v1[MAX], v2[MAX], v3[MAX];
#endif

int main(int argc, char** argv){
    int i;
    struct timespec cgt1,cgt2;
    double ncgt; //para tiempo de ejecución
  
    if(argc<2){
      printf("Faltan nº componentes del vector\n");
      exit(-1);
    }
    unsigned int N=atoi(argv[1]);
    #ifdef VECTOR_LOCAL
    double v1[N], v2[N], v3[N];
    
    #endif
    #ifdef VECTOR_GLOBAL
    if(N>MAX) N=MAX;
    #endif
    #ifdef VECTOR_DYNAMIC
    double *v1, *v2, *v3;
    v1= (double*) malloc(N*sizeof(double));
    v2= (double*) malloc(N*sizeof(double));
    v3= (double*) malloc(N*sizeof(double));
    if((v1==NULL) || (v2==NULL) || (v3==NULL)){
      printf("Error en la reserva de espacio para los vectores\n");
      exit(-2);
    }
    #endif
  
    //Inicializar vectores 
    #pragma omp parallel private (i)
    { 
      for(i=0;i<N;i++)
      {	    
	#pragma omp sections //Por un lado inicializamos el vector
	  {
	   #pragma omp section
	   {
	      v1[i]= N*0.1+i*0.1; v2[i]=N*0.1-i*0.1; //los valores dependen de N
	      printf("Inicialización: La hebra %d ejecuta iteración %d\n",omp_get_thread_num(),i);
	   }
	  }
      }
    }
      double a= omp_get_wtime();
    //Calcular suma de vectores
    #pragma omp parallel private (i)
    {	
		for(i=0;i<N;i++){
		#pragma omp sections //Por otro calculamos la suma
		{
		  #pragma omp section
		  {
		    v3[i]=v1[i] + v2[i];
		    printf("Suma: La hebra %d ejecuta iteración %d\n",omp_get_thread_num(),i);
		  }
	      }
	    
	    ncgt=(double) (cgt2.tv_sec-cgt1.tv_sec)+ (double) ((cgt2.tv_nsec-cgt1.tv_nsec)/(1.e+9));
	}
    }
    //Imprimir resultado de la suma y el tiempo de ejecución
    #ifdef PRINTF_ALL
    printf("Tiempo(seg.): %11.9f\t / Tamaño Vectores:%u\n",omp_get_wtime()/*ncgt*/,N);
      for(i=0;i<N;i++){
	printf("thread %d ejecuta la iteración %d del bucle\n",omp_get_thread_num(),i);
	printf("Elapsed time: %11.9f\t\n",omp_get_wtime()-a);
	printf("/V1[%d]+V2[%d](%8.6f+%8.6f=%8.6f)/\n", i,i,i,v1[i],v2[i],v3[i]);
      }
    #else
    printf("Tiempo: %11.9f\t / Tamaño Vectores:%u\n",
	   ncgt,N,v1[0],v2[0],v3[0],N-1,N-1,N-1,v1[N-1],v2[N-1],v3[N-1]);
    #endif
    
    #ifdef VECTOR_DYNAMIC
      free(v1); //libera el espacio reservado para v1
      free(v2); //libera el espacio reservado para v2
      free(v3); //libera el espacio reservado para v3
    #endif
    return 0;
}