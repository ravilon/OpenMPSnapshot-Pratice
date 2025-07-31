#include <sys/time.h>
#include "include/data_management.h"
#include "include/divisors.h"

int main (int argc, char **argv)
{

    if (argc != 2)
    {
        fprintf(stderr, "Incorrect number of arguments.\n");
        fprintf(stderr, "Usage: %s <path/to/file>\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    char *file_name = argv[1];
    int input_size = get_input_size(file_name);

    int *values = (int *) malloc (input_size * sizeof(int));
    read_file(file_name, values, input_size);

    // Could also store the results overwriting the input array
    int *results = (int *) malloc (input_size * sizeof(int));

    // clock_t counts CPU time between all cores (threads)
    int i;
    struct timeval begin, end;
    gettimeofday(&begin, NULL);

    #pragma omp parallel for private(i) shared(values, input_size, results) num_threads(N_THREADS)
    for (i = 0; i < input_size; i++)
        results[i] = count_divisors(values[i]);

    gettimeofday(&end, NULL);

    double total_time = (end.tv_sec - begin.tv_sec) + (end.tv_usec - begin.tv_usec) / 1000000.0;

    write_file(results, input_size);
    printf("Processing time: %0.3lfs\n", total_time);

    free(results);
    free(values);
    return 0;
}
