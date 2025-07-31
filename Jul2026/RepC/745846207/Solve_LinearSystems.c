/*
** Questo file contiene le funzioni per risolvere un sistema di equazioni lineari NxN utilizzando la decomposizione LU.
** L'implementazione si basa sul materiale teorico, reperito in rete, contenuto in:
**
**      - "Numerical Methods - A Software Approach" di Johnston (1982), pp.28-44
**      - "The C Programming Language" di Kernighan & Ritchie (1978), p.104
**      - "An efficient implementation of LU decomposition in C" di A. Meyer (1988)
**      - "Parallel Scientific Computation: A structured approach using BSP and MPI" di R.H. Bisseling (2004)
**
** Le funzioni:
**
**      LUDecompose --> esegue la fase di decomposizione.
**      LUSolve --> risolve un sistema dato un vettore b dei termini noti.
**
*/

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include <stdlib.h>


static int *pivot = NULL;       // Vettore Pivot
#define SMALLEST_PIVOT 1.0e-5  // Più piccolo pivot non-singolare consentito



/*
** Questa funzione esegue la fase di decomposizione LU. La matrice dei coefficienti in input verrà sovrascritta
** con le matrici Lu (Low & Upper) computate nel processo.
**
**  Parametri :
**
**  mat - È la matrice dei coefficienti. Questa funzione esegue la lettura per riga per riga:
**        si dichiara la matrice come un puntatore ad un array monodimensionale. Non si utilizza l'inefficiente
 *        implementazione di vettore bidimensionale ( double mat[SIZE][SIZE] ), come già fornito dal linguaggio.
**
**  n -   Questo parametro indica la dimensione del sistema corrente (NxN).
**
*/

int LUDecompose(double *mat, int n) {


    // Setto il numero di core della CPU usati per parallelizzare la "LU Decomposizione":

    int numCores = omp_get_num_procs();
    omp_set_num_threads(numCores);

    //#################################################################################//


    int *pivot = (int*)malloc(n * sizeof(int));  // Alloca memoria per il vettore pivot

    for (int k = 0; k < n - 1; k++) {

        // Inizializza il pivot
        for (int i = 0; i < n; i++) {
            pivot[i] = i;
        }

        // Cerca la riga pivot col valore massimo
        for (int i = k + 1; i < n; i++) {
            if (fabs(mat[i * n + k]) > fabs(mat[pivot[i] * n + k])) {
                pivot[i] = k;
            }
        }

        // Verifica la singolarità (pivot troppo prossimo allo zero!)
        if (fabs(mat[pivot[k] * n + k]) < SMALLEST_PIVOT) {
            fprintf(stderr, "Errore in LUDecompose - La matrice è singolare !!!\n");
            free(pivot);  //Libera la memoria allocata
            return -1;
        }

        // Scambia righe se la riga pivot non è la riga corrente
        #pragma omp parallel for
        for (int j = 0; j < n; j++) {
            double temp = mat[k * n + j];
            mat[k * n + j] = mat[pivot[k] * n + j];
            mat[pivot[k] * n + j] = temp;
        }

        // Decomponi la matrice ( LU Decomposition - Gauss Elimination Method (1826) )
        #pragma omp parallel for
        for (int i = k + 1; i < n; i++) {
            double factor = mat[i * n + k] / mat[k * n + k];
            for (int j = k; j < n; j++) {
                mat[i * n + j] -= factor * mat[k * n + j];
            }
            mat[i * n + k] = factor;
        }


    }


    free(pivot);

    return numCores;

}



/*
**  Questa funzione risolve un sistema di equazioni lineari, dato il vettore dei termini noti.
**  La Matrice dei Coefficienti va prima decomposta in forma LU, e solo poi si può risolvere!
**
**    Parametri:
**
**    mat - E' la LU Decomposizione della matrice dei coefficienti iniziale.
**          La matrice originale va prima passata alla funzione di decomposizione
**          per essere risolta.
**
**    b -   E' il vettore dei termini noti, questa funzione computa il vettore dei
**          risultati e lo ritorna sovrascrivvendolo in b.
**
**    n -   Questo parametro indica la dimensione del sistema corrente (NxN).
**
**    size - Numero totale di processi MPI che partecipano all'esecuzione .
**
**    rank - Identificatore unico assegnato a ciascun processo.
**
*/

void LUSolve(double *mat, double *b, int n, int size, int rank) {

    // Forward substitution con pivoting parziale:
    for (int k = 0; k < n - 1; k++) {

        //Suddivisione del lavoro tra i processi
        if (k % size == rank) {

            for (int i = k + 1; i < n; i++) {
                b[i] -= mat[i * n + k] * b[k];
            }

        }

        //Invio del risultato a tutti i processi
        MPI_Bcast(b, n, MPI_DOUBLE, k % size, MPI_COMM_WORLD);

    }



    // Backward substitution con pivoting parziale:
    for (int k = n - 1; k >= 0; k--) {

        double sum = 0.0;

        //Suddivisione del lavoro tra i processi
        if (k % size == rank) {

            for (int j = k + 1; j < n; j++) {
                sum += mat[k * n + j] * b[j];
            }

            b[k] = (b[k] - sum) / mat[k * n + k];
        }

        //Invio del risultato a tutti i processi
        MPI_Bcast(b, n, MPI_DOUBLE, k % size, MPI_COMM_WORLD);

    }


}