#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif /* _OPENMP */

#define MAX_LINE_LENGTH 80

int read_file(char *path, int *buff, uint32_t *size) {
    char line[MAX_LINE_LENGTH] = {0};
    unsigned int line_count = 0, i = 0;

    /* Open file */
    FILE *file = fopen(path, "r");

    if (!file) {
        perror(path);
        return EXIT_FAILURE;
    }

    /* Get each line until there are none left */
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        /* Print each line */
        // printf("line[%06d]: %s", ++line_count, line);
        buff[i] = atoi(line);
        i++;
    }

    *size = i;
    /* Close file */
    if (fclose(file)) {
        return EXIT_FAILURE;
        perror(path);
    }
    return 0;
}

int main(int argc, char *argv[]) {
    uint32_t num_size, true_n0 = 646016;
    int numbers[2000000];

    read_file("num.txt", numbers, &num_size);
    printf("Size of integer array/file: %d\n", num_size);

    // first loop
    int maxval = 0;
// #pragma omp parallel for reduction(max: maxval)
#pragma omp parallel for
    for (uint32_t i = 0; i < num_size; i++) {
        if (numbers[i] > maxval) {
#pragma omp critical
            maxval = numbers[i];
#ifdef _OPENMP
            // printf("%d\n", omp_get_thread_num());
#endif /* _OPENMP */
        }
    }
    printf("max number in file: %d\n", maxval);

    // second loop
    int num_n0 = 0;
    // NOTE: (armin) either use reduction or atomic
// #pragma omp parallel for reduction(+ : num_n0)
#pragma omp parallel for
    for (uint32_t i = 0; i < num_size; i++) {
        if (numbers[i] == 0) {
#ifdef _OPENMP
            // printf("%d\n", omp_get_thread_num());
#endif /* _OPENMP */
#pragma omp atomic
            num_n0++;
        }
    }
    printf("number of 0s in file: %d\n", num_n0);
    printf("true number of 0s in file: %d\n", true_n0);

    return 0;
}
