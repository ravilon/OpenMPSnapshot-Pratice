#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void como_usar(char prog_name[]);

void get_argumetnos(char* argv[], int* bin_count_p, float* min_meas_p,
                    float* max_meas_p, int* data_count_p, int* thread_count_p);

void gerar_dados(float min_meas, float max_meas, float data[], int data_count);

void Gen_bins(float min_meas, float max_meas, float bin_maxes[],
              int bin_counts[], int bin_count);

int qual_bin(float data, float bin_maxes[], int bin_count, float min_meas);

void printar_histograma(float bin_maxes[], int bin_counts[], int bin_count,
                        float min_meas);

int main(int argc, char* argv[]) {
    int bin_count, i, bin;
    float min_meas, max_meas;
    float* bin_maxes;
    int *bin_counts, *loc_bin_counts;
    int data_count;
    float* data;
    int thread_count;

    if (argc != 6) como_usar(argv[0]);
    get_argumetnos(argv, &bin_count, &min_meas, &max_meas, &data_count,
                   &thread_count);

    bin_maxes = malloc(bin_count * sizeof(float));
    bin_counts = malloc(bin_count * sizeof(int));
    loc_bin_counts = malloc(thread_count * bin_count * sizeof(int));
    data = malloc(data_count * sizeof(float));

    gerar_dados(min_meas, max_meas, data, data_count);

    Gen_bins(min_meas, max_meas, bin_maxes, bin_counts, bin_count);
    memset(loc_bin_counts, 0, thread_count * bin_count * sizeof(int));

#pragma omp parallel num_threads(thread_count) private(bin, i)
    {
        int j, my_rank = omp_get_thread_num();
        int my_offset = my_rank * bin_count;

#pragma omp for
        for (i = 0; i < data_count; i++) {
            my_rank = omp_get_thread_num();
            bin = qual_bin(data[i], bin_maxes, bin_count, min_meas);
            loc_bin_counts[my_offset + bin]++;
        }

#pragma omp barrier

#pragma omp for
        for (i = 0; i < bin_count; i++)
            for (j = 0; j < thread_count; j++) {
                bin_counts[i] += loc_bin_counts[j * bin_count + i];
            }
    }

    printar_histograma(bin_maxes, bin_counts, bin_count, min_meas);

    free(data);
    free(bin_maxes);
    free(bin_counts);
    free(loc_bin_counts);
    return 0;
}

void como_usar(char prog_name[]) {
    fprintf(stderr, "usage: %s <bin_count> <min_meas> ", prog_name);
    fprintf(stderr, "<max_meas> <data_count> <thread_count>\n");
    exit(0);
}

void get_argumetnos(char* argv[], int* bin_count_p, float* min_meas_p,
                    float* max_meas_p, int* data_count_p, int* thread_count_p) {
    *bin_count_p = strtol(argv[1], NULL, 10);
    *min_meas_p = strtof(argv[2], NULL);
    *max_meas_p = strtof(argv[3], NULL);
    *data_count_p = strtol(argv[4], NULL, 10);
    *thread_count_p = strtol(argv[5], NULL, 10);
}

void gerar_dados(float min_meas, float max_meas, float data[], int data_count) {
    int i;

    srandom(0);
    for (i = 0; i < data_count; i++)
        data[i] =
            min_meas + (max_meas - min_meas) * random() / ((double)RAND_MAX);
}

void Gen_bins(float min_meas, float max_meas, float bin_maxes[],
              int bin_counts[], int bin_count) {
    float bin_width;
    int i;

    bin_width = (max_meas - min_meas) / bin_count;

    for (i = 0; i < bin_count; i++) {
        bin_maxes[i] = min_meas + (i + 1) * bin_width;
        bin_counts[i] = 0;
    }
}

int qual_bin(float data, float bin_maxes[], int bin_count, float min_meas) {
    int bottom = 0, top = bin_count - 1;
    int mid;
    float bin_max, bin_min;

    while (bottom <= top) {
        mid = (bottom + top) / 2;
        bin_max = bin_maxes[mid];
        bin_min = (mid == 0) ? min_meas : bin_maxes[mid - 1];
        if (data >= bin_max)
            bottom = mid + 1;
        else if (data < bin_min)
            top = mid - 1;
        else
            return mid;
    }

    /* Whoops! */
    fprintf(stderr, "Data = %f doesn't belong to a bin!\n", data);
    fprintf(stderr, "Quitting\n");
    exit(-1);
}

void printar_histograma(float bin_maxes[], int bin_counts[], int bin_count,
                        float min_meas) {
    int i, j;
    float bin_max, bin_min;

    for (i = 0; i < bin_count; i++) {
        bin_max = bin_maxes[i];
        bin_min = (i == 0) ? min_meas : bin_maxes[i - 1];
        printf("%.3f-%.3f:\t", bin_min, bin_max);
        for (j = 0; j < bin_counts[i]; j++) printf("X");
        printf("\n");
    }
}