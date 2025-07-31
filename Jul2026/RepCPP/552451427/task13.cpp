#include <iostream>
#include <omp.h>

using namespace std;

const int SIZE = 30;

int main() {
        
    int a[SIZE] = {1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1};

    int num = 0;

    #pragma omp parallel for reduction(+: num)
    for (int i = 0; i < SIZE; i++) {
        if (a[i] == 1) {
            num = num + (1 << SIZE - i - 1);
        }
    }

    printf("Result: \n");
    for (int i = 0; i < SIZE; i++) {
        printf("%d", a[i]);
    }
    printf(" in binary \n= %d in decimal.\n", num);

    return 0;
}