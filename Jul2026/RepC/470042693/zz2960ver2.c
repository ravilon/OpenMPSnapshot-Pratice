#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define MAX_BIN_NUM 50
#define MAX_THREAD_NUM 100

void print_help(char *executable);

int main(int argc, char *argv[])
{
    // Command line arguments processing
    char *executable = argv[0];
    if (argc != 4)
    {
        printf("Error: invalid arguments\n\n");
        print_help(executable);
        return 1;
    }
    char *bin_count_str = argv[1];
    char *thread_count_str = argv[2];
    char *file_name = argv[3];

    // Open input file
    FILE *fp = fopen(file_name, "r");

    if (fp == NULL)
    {
        printf("Error: cannot create file %s\n", file_name);
        return 1;
    }

    int bin_count = atoi(bin_count_str);
    int thread_count = atoi(thread_count_str);

    if (bin_count <= 0 || bin_count > MAX_BIN_NUM)
    {
        printf("Error: invalid bin count %s\n", bin_count_str);
        return 1;
    }

    if (thread_count <= 0 || thread_count > MAX_THREAD_NUM)
    {
        printf("Error: invalid thread count %s\n", thread_count_str);
        return 1;
    }

    // Read num count in input file
    int num_count = 0;
    fscanf(fp, "%d", &num_count);

    // Read in numbers
    double *nums = (double *)malloc(num_count * sizeof(double)); // use malloc to prevent segment fault when too many numbers
    for (int i = 0; i < num_count; i++)
    {
        fscanf(fp, "%lf", &nums[i]);
    }

    fclose(fp);

    // Intialize bin_counter array
    int bin_counter[bin_count];
    for (int i = 0; i < bin_count; i++)
    {
        bin_counter[i] = 0;
    }

    // Calculate the range length of indexes for numbers to be processed in per thread
    int num_range_length = (int)ceil((double)num_count / thread_count);

    double start_time, finish_time;
    start_time = omp_get_wtime(); // record start time

#pragma omp parallel for num_threads(thread_count) \
    shared(bin_counter)
    for (int i = 0; i < thread_count; i++)
    {
        // Intialize local_bin_counter array
        int local_bin_counter[bin_count];
        for (int i = 0; i < bin_count; i++)
        {
            local_bin_counter[i] = 0;
        }

        // Calculate number range for this thread
        int start = i * num_range_length;
        int end = i * num_range_length + num_range_length;

        // Go through all numbers in assigned range
        for (int j = start; j < end && j < num_count; j++)
        {
            // Find bin index for number j
            int bin_index = (int)(nums[j] * bin_count / 100.0);
            // Increase the local counter
            local_bin_counter[bin_index]++;
        }

        // Increase the global counter
        for (int j = 0; j < bin_count; j++)
        {
            bin_counter[j] += local_bin_counter[j];
        }
    }

    finish_time = omp_get_wtime();

    // Print out result
    for (int i = 0; i < bin_count; i++)
    {
        printf("bin[%d]=%d\n", i, bin_counter[i]);
    }

    // Print time statistics
    printf("Parallel part finished in %lf sec.\n", finish_time - start_time);
}

void print_help(char *executable)
{
    printf("usage: %s b t filename\n\n", executable);
    printf("A parallel version of histagram statistics counter where each thread is responsible for a subset of the numbers.\n\n");
    printf("positional arguments:\n");
    printf("  b          the number of bins, 0 < b <= %d\n", MAX_BIN_NUM);
    printf("  t          the number of threads, 0 < t <= %d\n", MAX_THREAD_NUM);
    printf("  filename   the name of the text file that contains the floating point numbers\n");
}
