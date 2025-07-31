#include <omp.h>
#include <algorithm>
#include <cstdio>

void quickSort(unsigned* a, unsigned n) {
    long i = 0, j = n-1;
    unsigned pivot = a[n / 2];
    do {
        while (a[i] < pivot) i++;
        while (a[j] > pivot) j--;
        if (i <= j) {
            std::swap(a[i], a[j]);
            i++; j--;
        }
    } while (i <= j);
    if (j > 0) quickSort(a, j+1);
    if (n > i) quickSort(a + i, n - i);
}

void quickSort_OMP(unsigned* a, unsigned n) {
    long i = 0, j = n-1;
    unsigned pivot = a[n / 2];
    do {
        while (a[i] < pivot) i++;
        while (a[j] > pivot) j--;
        if (i <= j) {
            std::swap(a[i], a[j]);
            i++; j--;
        }
    } while (i <= j);
    #pragma omp task shared(a)
    {
        if (j > 0) quickSort_OMP(a, j+1);
    }
    #pragma omp task shared(a)
    {
        if (n > i) quickSort_OMP(a + i, n - i);
    }
    #pragma omp taskwait
}

void randomize(unsigned* arr, unsigned N) {
    for (unsigned i = 0; i < N; i++)
        arr[i] = std::rand();
}

void printresult(unsigned* arr, unsigned N) {
    for (unsigned i = 0; i < N; i++)
        std::printf("%u ", arr[i]);
}

int main() {
    unsigned N = 10000000;
    unsigned* arr0 = new unsigned[N];
    unsigned* arr = arr0;
    randomize(arr0, N);

    std::printf("QS\n");
    double t0 = omp_get_wtime();
    quickSort(arr, N);
    std::printf("time: %f\n", omp_get_wtime() - t0);
    //printresult(arr, N);

    std::printf("\nQS_OMP\n");
    arr=arr0;
    t0 = omp_get_wtime();
    quickSort_OMP(arr, N);
    std::printf("time: %f\n", omp_get_wtime() - t0);
    //printresult(arr, N);
    return 0;
}