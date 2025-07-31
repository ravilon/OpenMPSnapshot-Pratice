#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

void no_reduction() {
    float a[] = {4.0, 3.0, 3.0, 1000.0};

    int i;
    float sum = 0.0;
    for (i = 0; i < 4; i++) sum += a[i];
    printf("sum (no reduction) = %4.1f\n", sum);
}

void reduction() {
    float a[] = {4.0, 3.0, 3.0, 1000.0};

    int i;
    float sum = 0.0;

#pragma omp parallel for num threads(2) reduction(+ : sum)
    for (i = 0; i < 4; i++) sum += a[i];
    printf("sum (with reduction) = %4.1f\n", sum);
}

int main(void) {
    no_reduction();
    reduction();

    return 0;
}
