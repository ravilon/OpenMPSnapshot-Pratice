#include <mpi.h>
#include "include/data_management.h"
#include "include/divisors.h"

#define MASTER 0

void help (char **argv)
{
    fprintf(stderr, "Incorrect number of arguments.\n");
    fprintf(stderr, "Usage: %s <path/to/file>\n", argv[0]);
}

int main (int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc != 2)
    {
        if (rank == 0)
            help(argv);
        goto mpi_end;
    }

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int input_size;
    int *input_data = NULL;
    int *each_proc_buf = NULL;
    int *sendcounts = NULL;
    int *displs = NULL;

    if (rank == MASTER)
    {
        // Input data reading
        char *file_name = argv[1];
        input_size = get_input_size(file_name);
        input_data = (int *) malloc (input_size * sizeof(int));
        read_file(file_name, input_data, input_size);

        sendcounts = (int *) malloc (size * sizeof(int));
        displs = (int *) malloc (size * sizeof(int));
        int remaining = input_size;

        // Calculates sendcounts and displacements for potentially uneven scattering accordingly to input data length
        for (int i = 0; i < size; i++)
        {
            sendcounts[i] = remaining / (size - i);
            displs[i] = i > 0 ? displs[i - 1] + sendcounts[i - 1] : 0;
            remaining -= sendcounts[i];
        }
    }

    // Sends the array size that each process will work
    int local_size;
    MPI_Scatter(sendcounts, 1, MPI_INT, &local_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocates memory for the portions that each process will operate
    each_proc_buf = (int *) malloc (local_size * sizeof(int));

    // All processes must be at the same stage to start time measuring
    MPI_Barrier(MPI_COMM_WORLD);
    double begin = MPI_Wtime();

    // TODO: Scatter only to the slave processes. Master shouldn't do processing (only I/O)
    // Splits the data between all the processes (even if it's not equally divisible)
    MPI_Scatterv(input_data, sendcounts, displs, MPI_INT, each_proc_buf, local_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculates the number of divisors for each value from the portion received by each process
    int i;
    #pragma omp parallel for private(i) shared(each_proc_buf, local_size) num_threads(N_THREADS)
    for (i = 0; i < local_size; i++)
        each_proc_buf[i] = count_divisors(each_proc_buf[i]);

    // Gathers all the calculated data for the master process
    MPI_Gatherv(each_proc_buf, local_size, MPI_INT, input_data, sendcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    // All processes must be here to stop time measuring
    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    // The master process prints and stores the gathered results
    if (rank == MASTER)
    {
        printf("Processing time: %0.3lfs\n", end - begin);
        write_file(input_data, input_size);

        // Clean up
        free(input_data);
        free(sendcounts);
        free(displs);
    }
    free(each_proc_buf);

mpi_end:
    MPI_Finalize();

    return 0;
}
