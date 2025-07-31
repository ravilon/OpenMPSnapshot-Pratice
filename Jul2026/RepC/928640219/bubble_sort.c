#include <omp.h>
#include <stdio.h>

#define N 10

void parallel_bubble_sort(int v[]) {
    for (int i = 0; i < N; i++) {
        // Fase par
        #pragma omp parallel for
        for (int j = 0; j < N - 1; j += 2) {
            if (v[j] > v[j + 1]) {
                int temp = v[j];
                v[j] = v[j + 1];
                v[j + 1] = temp;
            }
        }

        // Fase ímpar
        #pragma omp parallel for
        for (int j = 1; j < N - 1; j += 2) {
            if (v[j] > v[j + 1]) {
                int temp = v[j];
                v[j] = v[j + 1];
                v[j + 1] = temp;
            }
        }
    }
}

int main() {
    int v[N] = {3, 5, 12, 35, 61, 12, 7, 43, 1, 0};

    parallel_bubble_sort(v);

    printf("Vetor ordenado:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", v[i]);
    }
    printf("\n");
    return 0;
}
