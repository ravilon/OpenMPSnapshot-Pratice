/*
Curso: Matrices Distribuidas
Fecha: 18/11/2019
Programadores:
    Castro Alexander
    Palomino Octavio
    Tapias John
*/
/*
Compilar:
    gcc -o kmeans -fopenmp -lm kmeans.c
Establecer numero de threads:
    export OMP_NUM_THREADS=4
Ejecutar:
    ./kmeans -c 6 -i 8 -a datos_muestra_2.txt
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

// ---- Constantes
#define NUM_ATRIBUTOS 4
#define NUM_ID 9
#define ROOT 0
typedef enum { false, true } bool;

// ---- Estructuras
struct Cliente
{
    char id[NUM_ID];
    float diasMora;
    float lineaCredito;
    float anioCastigo;
    float anioUltimoPago;
    int grupo;
};
struct Centroide
{
    int numElementos;
    float diasMora;
    float lineaCredito;
    float anioCastigo;
    float anioUltimoPago;
};


// ---- Funciones
void validarParametros(int argc, char **argv, int *numCentroides, char **ruta, int *iterMax);
void obtenerNumeroObservaciones(char *rutaArchivo, int *observaciones);
void leerArchivo(char *rutaArchivo, int *observaciones);
void leerClientes(struct Cliente* clientes, int observaciones, char *rutaArchivo);
void mostrarClientes(struct Cliente clientes[], int observaciones);
void estandarizarDatos(struct Cliente *clientes, int observaciones);
void inicializarCentroides(struct Cliente *clientes, int observaciones, int numCentroides);
void calcularCentroides(struct Cliente *clientes,
                        int inicio_hilo,
                        int num_obs_hilo,
                        struct Centroide *centroides, 
                        int numCentroides);
void mostrarCentroides(struct Centroide *centroides, int numCentroides);
float calcularDistancia(struct Cliente cliente, struct Centroide centroide);

// ---- Función Main
int main(int argc, char **argv) {

    // ---- Declaracion de variables
    int i = 0, j, k;
    int numCentroides, numObservaciones;
    float *distancias; 
    char *rutaArchivo;
    struct Cliente *clientes;
    struct Centroide *centroides;
    bool termina = false;
    bool debug = false;
    int iterMax;
    int my_rank, num_hilos, num_obs_hilo, inicio_hilo;

    // ---- Establecer numero de centroides
    validarParametros(argc, argv, &numCentroides, &rutaArchivo, &iterMax);
    // ---- Captura del total de registros ---- 
    obtenerNumeroObservaciones(rutaArchivo, &numObservaciones);
    // ---- Reservar memoria clientes
    clientes = (struct Cliente *)malloc(sizeof(struct Cliente) * numObservaciones);
    leerClientes(clientes, numObservaciones, rutaArchivo);
    // mostrarClientes(clientes, numObservaciones);
    estandarizarDatos(clientes, numObservaciones);
    // mostrarClientes(clientes, numObservaciones);

    // ---- Algoritmo k-means
    // Reservo memoria para los k centroides
    centroides = (struct Centroide *)malloc(sizeof(struct Centroide) * numCentroides);
    inicializarCentroides(clientes, numObservaciones, numCentroides);
    calcularCentroides(clientes, ROOT, numObservaciones, centroides, numCentroides);
    printf("\n ------------ CLIENTES: INICIO\n");
    mostrarClientes(clientes, numObservaciones);
    printf("\n ------------ CENTROIDES: INICIO\n");
    mostrarCentroides(centroides, numCentroides);

    // ----- Iteracion
    // Reservo memoria para las k distancias
    // distancias = (int *)calloc(numCentroides, sizeof(int));
    distancias = (float *)malloc(sizeof(float) * numCentroides);
    while (!termina) {
        if (i == iterMax) {
            printf("\n----------> FIN ITERACION\n", i+1);
        } else {
            printf("\n----------> ITERACION #%d\n", i+1);
        }

        #pragma openmp parallel default(shared) private(my_rank, num_hilos, num_obs_hilo, inicio_hilo)
        {
            // Identificador del hilo actual
            my_rank = omp_get_thread_num();
            // Numero total de hilos
            num_hilos =  omp_get_num_threads();
            // Bloque de observaciones por hilo
            num_obs_hilo = numObservaciones / num_hilos;
            inicio_hilo = num_obs_hilo * my_rank;
            // Ultimo hilo
            if (my_rank == num_hilos-1) {
                // El ultimo hilo puede tener mas observaciones
                num_obs_hilo = numObservaciones - inicio_hilo;
            }

            for (j=0; j < num_obs_hilo; j++) {
                // Calcular distancia entre cada observacion y cada centroide
                for (k=0; k < numCentroides; k++) {
                    distancias[k] = calcularDistancia(clientes[inicio_hilo+j], centroides[k]);
                    if (debug) {
                        printf("\nDistancia Cliente #%d\tCentroide #%d: %f\n", inicio_hilo+j, k, distancias[k]);
                    }
                }
                // Seleccionar la distancia menor
                int menor = 0;
                for (k=1; k < numCentroides; k++) {                
                    if (distancias[k] < distancias[menor]) {
                        menor = k;
                    }
                }
                // Asignar observacion al nuevo grupo
                clientes[inicio_hilo+j].grupo = menor + 1;
            }
        }

        i++;
        if (i > iterMax) {
            termina = true;
        } else {
            // Calcular centroides
            #pragma openmp parallel default(shared) private(my_rank, num_hilos, num_obs_hilo, inicio_hilo)
            {
                // Identificador del hilo actual
                my_rank = omp_get_thread_num();
                // Numero total de hilos
                num_hilos =  omp_get_num_threads();
                // Bloque de observaciones por hilo
                num_obs_hilo = numObservaciones / num_hilos;
                inicio_hilo = num_obs_hilo * my_rank;
                // Ultimo hilo
                if (my_rank == num_hilos-1) {
                    // El ultimo hilo puede tener mas observaciones
                    num_obs_hilo = numObservaciones - inicio_hilo;
                }
                calcularCentroides(clientes, inicio_hilo, num_obs_hilo, centroides, numCentroides);                
            }
            // Sincronizacion de hilos
            #pragma omp barrier
            if (debug) {
                mostrarCentroides(centroides, numCentroides);
            }
            
        }
    }
    
    printf("\n ------------ CLIENTES: RESULTADO\n");
    mostrarClientes(clientes, numObservaciones);
    printf("\n ------------ CENTROIDES: RESULTADO\n");
    mostrarCentroides(centroides, numCentroides);
    // Fin de la ejecucion
    exit(0);
}

// ---- Implementacion de funciones
void validarParametros(int argc, char **argv, int *numCentroides, char **ruta, int *iterMax) {
    if (argc == 1) {
        // Si no se especifica un numero de centroides
        // Se establecen k=4 centroides por defecto
        *numCentroides = 4;
    } else if (argc == 7) {
        *numCentroides = atoi(argv[2]);
        *iterMax = atoi(argv[4]);
        *ruta = argv[6];
        printf("\n\nPrograma kmeans con k=%d centroides\n", *numCentroides);
        printf("\nProcesado el archivo %s\n\n\n",*ruta);
    } else {
        fprintf(stderr, "\n\t** Uso incorrecto del programa**\n");
        fprintf(stderr, "\t<programa> -c <numero_de_centroides> -i <numero_iteraciones_max> -a <nombre_archivo>\n\n");
        exit(1);
    }
}

void obtenerNumeroObservaciones(char *rutaArchivo, int *observaciones) {
    int ch = 0;
    *observaciones = 0;
    // Abrir el archivo en modo lectura
    FILE *archivo = fopen (rutaArchivo, "r");
    // ---- Validacion de apertura ----.
    if (archivo == NULL) { // si apunta null error nombre mal
        fprintf(stderr, "\n\t** Error al abrir el archivo **\n");
        fprintf(stderr, "\n\t** Valide su existencia en la ruta indicada **\n\n");
        exit(1);
    }

    //ch = getc(archivo);
    while((ch=fgetc(archivo))!=EOF) {
        if(ch == '\n') {
            *observaciones +=1;
        }
    }
    // Retomar el inicio del archivo
    rewind(archivo);
    // Cerrar archivo
    fclose(archivo);
    printf("\n\nNúmero de registros ob=%d\n", *observaciones);
}

void leerClientes(struct Cliente* clientes, int observaciones, char *rutaArchivo) {
    int ch = 0;
    int observacionActual = 0;

    // Abrir el archivo en modo lectura
    FILE *archivo = fopen (rutaArchivo, "r");
    // ---- Validacion de apertura ----.
    if (archivo == NULL) { // si apunta null error nombre mal
        fprintf(stderr, "\n\t** Error al abrir el archivo **\n");
        fprintf(stderr, "\n\t** Valide su existencia en la ruta indicada **\n\n");
        exit(1);
    }
    // Leer observaciones
    while (observacionActual < observaciones) {
        printf("\nLeyendo observación #%d\n", observacionActual);
        fscanf(archivo, "%s", clientes[observacionActual].id);
        // printf("clientes[%d].id -> %s\n", observacionActual, clientes[observacionActual].id);
        fscanf(archivo, "%f", &clientes[observacionActual].diasMora);
        // printf("clientes[%d].diasMora -> %.2f\n", observacionActual, clientes[observacionActual].diasMora);
        fscanf(archivo, "%f", &clientes[observacionActual].lineaCredito);
        // printf("clientes[%d].lineaCredito -> %.2f\n", observacionActual, clientes[observacionActual].lineaCredito);
        fscanf(archivo, "%f", &clientes[observacionActual].anioCastigo);
        // printf("clientes[%d].anioCastigo -> %.2f\n", observacionActual, clientes[observacionActual].anioCastigo);
        fscanf(archivo, "%f\n", &clientes[observacionActual].anioUltimoPago);
        // printf("clientes[%d].anioUltimoPago -> %.2f\n", observacionActual, clientes[observacionActual].anioUltimoPago);
        
        observacionActual++;
    }
    fclose(archivo);
    //printf("\n** Numero de clientes: %zu", sizeof(clientes)/sizeof(*clientes));
    //return clientes;
}

void mostrarClientes(struct Cliente clientes[], int observaciones) {
    int i;
    printf("\n");
    for (i = 0; i < observaciones; i++) {
        printf("\nObservacion #%d\n", i+1);
        printf("clientes[%d].id -> %s\n", i+1, clientes[i].id);
        printf("clientes[%d].diasMora -> %f\n", i+1, clientes[i].diasMora);
        printf("clientes[%d].lineaCredito -> %f\n", i+1, clientes[i].lineaCredito);
        printf("clientes[%d].anioCastigo -> %f\n", i+1, clientes[i].anioCastigo);
        printf("clientes[%d].anioUltimoPago -> %f\n", i+1, clientes[i].anioUltimoPago);
        printf("clientes[%d].grupo -> %d\n", i+1, clientes[i].grupo);
    }
    printf("\n");
}

void estandarizarDatos(struct Cliente *clientes, int observaciones) {
    int i;
    // Declaro Medias
    float medias[NUM_ATRIBUTOS];
    // Declaro Desviaciones estandar
    float desviaciones[NUM_ATRIBUTOS];

    // Inicializo Medias y Desviaciones
    #pragma omp parallel for
    for (i = 0; i < NUM_ATRIBUTOS; i++) {
        medias[i] = 0;
        desviaciones[i] = 0;
    }

    // ---- Calculo de las medias
    for (i = 0; i < observaciones; i++) {
        medias[0] += clientes[i].diasMora;
        medias[1] += clientes[i].lineaCredito;
        medias[2] += clientes[i].anioCastigo;
        medias[3] += clientes[i].anioUltimoPago;
    }

    for (i = 0; i < NUM_ATRIBUTOS; i++) {
        medias[i] /= (float)observaciones;
    }

    // ---- Calculo de las desviaciones
    #pragma omp parallel for
    for (i = 0; i < observaciones; i++) {
        desviaciones[0] += pow(clientes[i].diasMora - medias[0], 2);
        desviaciones[1] += pow(clientes[i].lineaCredito - medias[1], 2);
        desviaciones[2] += pow(clientes[i].anioCastigo - medias[2], 2);
        desviaciones[3] += pow(clientes[i].anioUltimoPago - medias[3], 2);
    }

    for (i = 0; i < NUM_ATRIBUTOS; i++) {
        desviaciones[i] = sqrt(desviaciones[i]/(float)observaciones);
    }

    // ---- Estandarizar
    #pragma omp parallel for
    for (i = 0; i < observaciones; i++) {
        clientes[i].diasMora = (clientes[i].diasMora - medias[0]) / desviaciones[0];
        clientes[i].lineaCredito = (clientes[i].lineaCredito - medias[1]) / desviaciones[1];
        clientes[i].anioCastigo = (clientes[i].anioCastigo - medias[2]) / desviaciones[2];
        clientes[i].anioUltimoPago = (clientes[i].anioUltimoPago - medias[3]) / desviaciones[3];
    }

    for (i = 0; i < NUM_ATRIBUTOS; i++) {
        printf("media[%d] -> %f\n", i+1, medias[i]);
        printf("desviaciones[%d] -> %f\n", i+1, desviaciones[i]);
    }
    
}

void inicializarCentroides(struct Cliente *clientes, int observaciones, int numCentroides) {
    int i;
    // Seed
    srand(time(NULL)); 
    // Asignar un grupo al azar a cada cliente
    #pragma omp parallel for
    for (i = 0; i < observaciones; i++) {
        // Grupo entre 1 y numCentroides
        clientes[i].grupo = (rand() % numCentroides) + 1;
    }
}

void calcularCentroides(struct Cliente *clientes,
                        int inicio_hilo,
                        int num_obs_hilo,
                        struct Centroide *centroides, 
                        int numCentroides) {
    int i;
    // Inicializar centroide
    for (i = 0; i < numCentroides; i++) {
        // Grupo entre 1 y numCentroides
        centroides[i].numElementos = 0;
        centroides[i].diasMora = 0;
        centroides[i].lineaCredito = 0;
        centroides[i].anioCastigo = 0;
        centroides[i].anioUltimoPago = 0;
    }
    // Calcular centroide para cada grupo
    for (i = 0; i < num_obs_hilo; i++) {
        // Acumular para cada componente del centroide
        centroides[clientes[inicio_hilo+i].grupo - 1].diasMora = centroides[clientes[inicio_hilo+i].grupo - 1].diasMora +  clientes[inicio_hilo+i].diasMora;
        centroides[clientes[inicio_hilo+i].grupo - 1].lineaCredito = centroides[clientes[inicio_hilo+i].grupo - 1].lineaCredito + clientes[inicio_hilo+i].lineaCredito;
        centroides[clientes[inicio_hilo+i].grupo - 1].anioCastigo = centroides[clientes[inicio_hilo+i].grupo - 1].anioCastigo + clientes[inicio_hilo+i].anioCastigo;
        centroides[clientes[inicio_hilo+i].grupo - 1].anioUltimoPago = centroides[clientes[inicio_hilo+i].grupo - 1].anioUltimoPago + clientes[inicio_hilo+i].anioUltimoPago;
        // Contar el numero de clientes en el grupo para calcular el promedio
        centroides[clientes[inicio_hilo+i].grupo - 1].numElementos = centroides[clientes[inicio_hilo+i].grupo - 1].numElementos + 1;
    }
    // Calcular el promedio para cada centroide
    for (i = 0; i < numCentroides; i++) {
        // Grupo entre 1 y numCentroides
        centroides[i].diasMora = centroides[i].diasMora / centroides[i].numElementos;
        centroides[i].lineaCredito = centroides[i].lineaCredito / centroides[i].numElementos;
        centroides[i].anioCastigo = centroides[i].anioCastigo / centroides[i].numElementos;
        centroides[i].anioUltimoPago = centroides[i].anioUltimoPago / centroides[i].numElementos;
    }
}

void mostrarCentroides(struct Centroide *centroides, int numCentroides) {
    int i;
    printf("\n");
    for (i = 0; i < numCentroides; i++) {
        printf("\nCentroide #%d\n", i+1);
        printf("centroides[%d].numElementos -> %d\n", i+1, centroides[i].numElementos);
        printf("centroides[%d].diasMora -> %f\n", i+1, centroides[i].diasMora);
        printf("centroides[%d].lineaCredito -> %f\n", i+1, centroides[i].lineaCredito);
        printf("centroides[%d].anioCastigo -> %f\n", i+1, centroides[i].anioCastigo);
        printf("centroides[%d].anioUltimoPago -> %f\n", i+1, centroides[i].anioUltimoPago);
    }
    printf("\n");
}

float calcularDistancia(struct Cliente cliente, struct Centroide centroide) {
    float distancia = 0;
    distancia += pow(cliente.diasMora - centroide.diasMora, 2);
    distancia += pow(cliente.lineaCredito - centroide.lineaCredito, 2);
    distancia += pow(cliente.anioCastigo - centroide.anioCastigo, 2);
    distancia += pow(cliente.anioUltimoPago - centroide.anioUltimoPago, 2);
    distancia = sqrt(distancia);
    
    return (float)distancia;
}