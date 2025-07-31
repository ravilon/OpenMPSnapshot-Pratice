/*
    nume:   Dumitrescu Alexandra  
    grupa:  343C1  
    tema:   Odd Even Transposition Sort  
    data:   Decembrie 2023

*/

#include "odd_even_transposition_helper.h"
#include <omp.h>

/* given number of elements in array */
int N;

/* initial array */
int *v;

/*
    Descriere solutie:

    1) Pornind de la solutia initala seriala, m-am folosit de paralel for pentru a paraleliza
    fiecare iteratie intr-o faza para/impara
*/

void solve_oets() {
    for(int k = 0; k < N; k++) {
        int j;
        /* even phase */
        if(k % 2 == 0) {
            #pragma omp parallel for private(j) shared(v, N)
            for(j = 0; j < N - 1; j += 2) {
                if(v[j] > v[j + 1]) {
                    swap(&v[j], &v[j + 1]);
                }
            }
        } else {
            /* odd phase */
            #pragma omp parallel for private(j) shared(v, N)
            for(j = 1; j < N - 1; j += 2) {
                if(v[j] > v[j + 1]) {
                    swap(&v[j], &v[j + 1]);
                }
            }
        }
    }  
}


int main(int argc, char **argv) {
    /* get initial time */
    double begin = omp_get_wtime();

    /* check correct arguments */
    if(argc != ARGC) {
        printf("Invalid parameters! Try: ./serial <file.in>");
        return -1;
    }

    /* read input */
    read_input_data(argv[1]);

    /* solve OETS */
    solve_oets();

    /* print result */
    print_output();

    /* free used memory */
    free_memory();

    /* compute execution time */
    double end = omp_get_wtime();
    double time_spent = (double)(end - begin);

    /* print execution time to stderr*/
    fprintf(stderr, "%f\n", time_spent);
}