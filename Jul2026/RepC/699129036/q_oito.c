#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    int n = 20;
    int a[n];
    int i;
    int thread_count = 10;

    # pragma omp parallel for \
    num_threads(thread_count) \
    default(none) \
    private(i) \
    shared(a, n)
    for (i = 0; i < n; i++)
        a[i] = (i * (i + 1)) / 2;

    for (i = 0; i < n; i++)
        printf("%d ", a[i]);

    exit(0);
}
