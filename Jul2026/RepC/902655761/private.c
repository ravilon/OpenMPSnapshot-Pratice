#include <stdio.h>
#include <omp.h>
#define n 8

int main() {
    int i = 0;
    int a = 99;
    int id = 0;
    int b = 888;

    #pragma omp parallel num_threads(4) private(id, a) shared(b)
    {
        id = omp_get_thread_num();
        a = id;

        #pragma omp for private(a)
        for (i = 0; i < n; i++) {
            a = a + 1;
            b += id;  // Manipulação de 'b' sem sincronização
            printf("id=%d \t a=%d \t b=%d \t for i=%d\n", id, a, b, i);
        }
    }

    printf("\nValor final de a: %d\n", a);
    printf("\nValor final de b: %d\n", b);

    return 0;
}