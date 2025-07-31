#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <limits.h> // For INT_MIN, to be used *MCS*
#include <time.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define ARRAY_SIZE 200000000

void generate_random_array(int *array, int size) {

    /*array[0] = -2;
    array[1] = 1;
    array[2] = 3;
    array[3] = -7;
    array[4] = 11;
    array[5] = -2;
    array[6] = -6;
    array[7] = 12;
    array[8] = -3;
    array[9] = -1;*/

    for (int i = 0; i < size; i++) {
        array[i] = rand() % 201 - 100;  // Random numbers between -100 and 100
    }
}

double solve_mcs(int *x, int n) {
    // For such a large array, prefer "heap"
    int *s = malloc(n * sizeof(int));
    int *m = malloc(n * sizeof(int));
    int *s_m = malloc(n * sizeof(int));
    int *indx = malloc(n * sizeof(int));

    if (!s || !m || !s_m || !indx) {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    /*
     * ## Fork-Join parallelism strategy
     * Calculate parallel *s*
     * Synchronize
     * Calculate parallel *m*
     * Synchronize
     * Calculate parallel *s_m*
     * Synchronize
     * Calculate parallel *indx*
     * Synchronize
     * Calculate parallel *MCS*
     * Synchronize
     */

    s[0] = x[0];
    int r = x[0];
    double start_time_for_s = omp_get_wtime(); // Timers can be used calculations only for parallel regions.
    /*
     * prefix-sum:  see pg. between *37-39* in *06_Patterns.pdf* also see pg. *24* in *08_Reduce_Scan.pdf*
     */
    #pragma omp parallel for reduction(inscan, +: r)
    for (int i = 1; i < n; i++) {
        r += x[i];
        #pragma omp scan inclusive(r)
        s[i] = r;
    }
    double end_time_for_s = omp_get_wtime();
    double calculation_time_for_s = end_time_for_s - start_time_for_s;

    m[0] = s[0];
    int t = x[0];
    double start_time_for_m = omp_get_wtime();
    /*
     * prefix-min
     */
    #pragma omp parallel for reduction(inscan, min: t)
    for (int i = 1; i < n; i++) {
        t = MIN(t, s[i]);
        #pragma omp scan inclusive(t)
        m[i] = t;
    }
    double end_time_for_m = omp_get_wtime();
    double calculation_time_for_m = end_time_for_m - start_time_for_m;

    double start_time_for_sm = omp_get_wtime();
    #pragma omp for
    /*
     * If s[i] and m[i] were calculated, then s_m is calculated.
     */
    for (int i = 1; i < n; i++) {
        s_m[i] = s[i] - m[i - 1];
    }
    double end_time_for_sm = omp_get_wtime();
    double calculation_time_for_sm = end_time_for_sm - start_time_for_sm;

    indx[0] = 0;
    int indx_val = 0;
    double start_time_for_indx = omp_get_wtime();
    #pragma omp parallel for reduction(inscan, max: indx_val)
    for (int i = 1; i < n; i++) {
        if (s[i] < m[i-1]) {
            indx_val = i;
        }
        #pragma omp scan inclusive(indx_val)
        indx[i] = indx_val;
    }
    double end_time_for_indx = omp_get_wtime();
    double calculation_time_for_indx = end_time_for_indx - start_time_for_indx;

    int MCS = INT_MIN;
    int end_index = 0;
    double start_time_for_mcs = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp for reduction(max: MCS)
        for (int i = 0; i < n; i++) {
            if (s_m[i] > MCS) {
                MCS = s_m[i];
            }
        }

        #pragma omp for reduction(+: end_index)
        for (int i = 0; i < n; i++) {
            if (MCS == s_m[i]) {
                end_index = i;
            }
        }
    }

    double end_time_for_mcs = omp_get_wtime();
    double calculation_time_for_mcs = end_time_for_mcs - start_time_for_mcs;

    int start_index = indx[end_index] + 1;

    printf("MCS value: %d, indices: [%d, %d]\n", MCS, start_index, end_index);

    free(s);
    free(m);
    free(s_m);
    free(indx);

    return end_time_for_s + end_time_for_m + end_time_for_sm + end_time_for_mcs;
}


int main() {
    //srand(time(NULL));
    int *x = malloc(ARRAY_SIZE * sizeof(int));
    if (!x) {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    generate_random_array(x, ARRAY_SIZE);

    printf("Testing with array size: %d\n\n", ARRAY_SIZE);

    double execution_time = solve_mcs(x, ARRAY_SIZE);
    printf("Calculation time: %.4f seconds\n\n", (execution_time/1000000));

    free(x);
}
