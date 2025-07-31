/*
    nume:   Dumitrescu Alexandra  
    grupa:  343C1  
    tema:   Odd Even Transposition Sort  
    data:   Decembrie 2023
*/
#include "odd_even_transposition_helper.h"
#include "mpi.h"
#include <omp.h>

/* MASTER task */
#define MASTER 0

#define MIN(a,b) (((a)<(b))?(a):(b))

/* number of tasks - can be varied for Tema3 */
int     NUM_TASKS;

/* given number of elements in array */
int     N;

/* initial array */
int     *v;

/* compute starting indexes for odd/even phases */
int     start_par;
int     start_impar;

/* rank */
int     rank;


/* MASTER process will keep track of all starting and ending indexes of all processes */
int     *start_even_idxs;
int     *start_odd_idxs;
int     *end_idxs;

/*
    Descriere generala:
    1) Pornind de la varianta de OpenMP am paralelizat for-urile in care fiecare proces isi modifica
    portiunea sa asiignata din vector.
*/
void solve_oets() {
    /* split the array equally and get start and end indexes */
    int start_idx   = rank * (double) N/NUM_TASKS;
    int end_idx     = MIN((rank + 1) * (double) N/NUM_TASKS, N);
    
    if(rank == MASTER) {
        start_even_idxs     = malloc(N * sizeof(int));
        start_odd_idxs   = malloc(N * sizeof(int));
        end_idxs           = malloc(N * sizeof(int));
        
        /* HIBRID - I used OpenMP's parallel for*/
        #pragma omp parallel for shared(start_even_idxs, start_odd_idxs, end_idxs) num_threads(2)
        for(int j = 1; j < NUM_TASKS; j++) {
            /* compute for each process its starting and ending indexes */
            int start_idx_j     = j * (double) N/NUM_TASKS;
            int end_idx_j       = MIN((j + 1) * (double) N/NUM_TASKS, N);

            /* compute for each process its starting indexes for even and odd phases */
            start_even_idxs[j]  = compute_start_even_idx(start_idx_j);
            start_odd_idxs[j]   = compute_start_odd_idx(start_idx_j);

            /* store the ending index */
            end_idxs[j] = end_idx_j;
        }
    }

    /* each process will compute its starting indexes for both odd / even phase */
    start_par = compute_start_even_idx(start_idx);
    start_impar = compute_start_odd_idx(start_idx);

    for(int i = 0; i < N; i++) {
        /* Master sends merged updates to all processes  */
        MPI_Bcast(v, N, MPI_INT, MASTER, MPI_COMM_WORLD);
        
        if(i % 2 == 0) {
            /* HIBRID - I used OpenMP's parallel for */
            /* compute updates on given block of elements */
            #pragma omp parallel for shared(v) num_threads(2)
            for(int j = start_par; j < end_idx; j += 2) {
                /* not allowed to pass vector limits */
                if(!in_vector_limits(j + 1, N)) {
                    continue;
                }
                    
                if(v[j] > v[j + 1]) {
                    swap(&v[j], &v[j + 1]);
                }
            }

            /* send updates to MASTER */
            if(rank != MASTER) {
                MPI_Send(v + start_par, end_idx != N ? end_idx - start_par + 1 : end_idx - start_par, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
            }
        } else {
            /* HIBRID - I used OpenMP's parallel for */
            /* compute updates on given block of elements */
            #pragma omp parallel for shared(v) num_threads(2)
            for(int j = start_impar; j < end_idx; j += 2) {
                /* not allowed to pass vector limits */
                if(!in_vector_limits(j + 1, N)) {
                    continue;
                }

                if(v[j] > v[j + 1]) {
                    swap(&v[j], &v[j + 1]);
                }                  
            }

            /* send updates to MASTER */
            if(rank != MASTER) {
                MPI_Send(v + start_impar, end_idx != N ? end_idx - start_impar + 1 : end_idx - start_impar, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
            }                
        }

        /* MASTER process receives updates from all processes and merges them */
        if(rank == MASTER) {
            for(int j = 1; j < NUM_TASKS; j++) {
                int dec = i % 2 == 0 ? start_even_idxs[j] : start_odd_idxs[j];
                MPI_Recv(v + dec, end_idxs[j] != N ? end_idxs[j] - dec + 1 : end_idxs[j] - dec, MPI_INT, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }  
}

int main(int argc, char **argv) {
    /* Initialization, get size and rank for each process */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &NUM_TASKS);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* used for getting start time */
    double start_time = MPI_Wtime();

    /* MASTER process will read input data */
    if(rank == MASTER) {
        read_input_data(argv[1]);
    }

    /* After reading data, MASTER will broadcast to all processes number of elements in array - N */
    MPI_Bcast(&N, 1, MPI_INT, MASTER, MPI_COMM_WORLD);

    /* Rest of the processes will wait for receiving N and will allocate memory for array v */
    if(rank != MASTER) {
        v = (int *) malloc(N * sizeof(int));
        if(!v) {
            printf("Bad malloc!");
            exit(-1);
        }
    }

    /* solve OETS */
    solve_oets();

    /* MASTER will print output and free allocated memory for special arrays */
    if(rank == MASTER) {
        print_output();
        free(start_odd_idxs);
        free(start_even_idxs);
        free(end_idxs);
    }

    /* free input array */
    free_memory();

    /* compute execution time */
    double end_time = MPI_Wtime();
    MPI_Finalize();

    if(rank == MASTER)
        fprintf(stderr, "%f\n", end_time - start_time);
}