#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define PRINTF_ALL
#define VECTOR_DYNAMIC //descomentar para que los vectores sean variables ...
		      //dinámicas (memoria reautilizable durante la ejecución)
#ifdef VECTOR_GLOBAL
#define MAX 33554432
double matriz[MAX], matriz2[MAX], resultado[MAX];
#endif
		      
		      
int main(int argc, char** argv){
    int i,j, temporal,k;
    struct timespec cgt1,cgt2;
    double ncgt; //para tiempo de ejecución
 
    if(argc<3){
      printf("Faltan nº componentes de las matrices <nº_filas_matriz_y_nº_columnas_matriz> o chunk\n");
      exit(-1);
    }
    unsigned int N=atoi(argv[1]);
    unsigned int chunk=atoi(argv[2]);
    omp_set_schedule(N,chunk); //modificamos run-sched-var
    
    int **matriz, *vector, *resultado;
    //Reservamos espacio pa la matriz
    //*******************************
    matriz= (int**) malloc(N*sizeof(int*));
    for(i=0;i<N;i++)
      matriz[i]=(int *) malloc((N-i)*sizeof(int)); //escalonamos la matriz
    
    //Reservamos memoria para los vectores    
    vector= (int*) malloc(N*sizeof(int));
    resultado=(int *) malloc(N*sizeof(int));    
    //*******************************
    if((matriz==NULL) || (vector==NULL) || (resultado==NULL)){
      printf("Error en la reserva de espacio para los vectores\n");
      exit(-2);
    }
  
    //Inicializar matrices
    #pragma parallel for
    for(i=0;i<N;i++){
      for(j=0;j<N-i;j++){
	  matriz[i][j]= i*j;
      }
    }
    
    //Inicializamos los vectores
    #pragma parallel for
    for(i=0;i<N;i++)
	vector[i]=i+10;
    
    #pragma parallel for  
    for(i=0;i<N;i++){
	  resultado[i]=0;
    }   
    //***********************
       
   
    clock_gettime(CLOCK_REALTIME,&cgt1);
    //Calcular multiplicación de la matrices
    //**************************************
#pragma omp parallel for firstprivate(temporal) lastprivate(temporal)schedule(guided,chunk)
    for(i=0;i<N;i++){
	resultado[i]=0;
	#pragma omp parallel for reduction(+:temporal)
	for(j=0;j<N-i;j++){
	  temporal+=matriz[i][j] * vector[i];
	  #pragma omp atomic
	  resultado[i]+=temporal;
	}
    }
    //**************************************
    clock_gettime(CLOCK_REALTIME,&cgt2);
    ncgt=(double) (cgt2.tv_sec-cgt1.tv_sec) + (double) ((cgt2.tv_nsec-cgt1.tv_nsec)/(1.e+9));
        
    #ifdef PRINTF_ALL
    printf("Tiempo(seg.): %11.9f\t / Tamaño Vectores:%u\n",ncgt,N);
    /*
    for(i=0;i<N;i++){
      for(j=0;j<N-i;j++)
	printf("/matriz[%d][%d]*vector[%d](%d*%d=%d)/\n", i,j,i,matriz[i][j],vector[i],matriz[i][j] * vector[i]);
    }
    printf("Resultado final resultante:\n");
    for(i=0;i<N;i++){
      for(j=0;j<N-i;j++)
	printf("resultado[%d]= %d\n", i,resultado[i]);
    }
    */
    #else
    printf("Tiempo(seg.): %11.9f\t / Tamaño Vectores:%u\n",
	   ncgt,N,matriz[0][0],vector[0],resultado[0],N-1,N-1,N-1,matriz[N-1][N-1],vector[N-1],resultado[N-1]);
    #endif
    
    #ifdef VECTOR_DYNAMIC
      free(matriz); //libera el espacio reservado para v1
      free(vector); //libera el espacio reservado para v2
      free(resultado); //libera el espacio reservado para v3
    #endif
    return 0;
}