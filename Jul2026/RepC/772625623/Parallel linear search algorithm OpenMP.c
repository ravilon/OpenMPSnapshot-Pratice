#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 100
int main() {
    int key = 42;
    int count = 0;

    int arr[N];
    for (int i = 0; i < N; i++) {
        arr[i] = rand() % 100;
    }
    for(int i=0;i<N;i++) //print the randomly filled array
    {
        printf("%d ",arr[i]);
    }
    #pragma omp parallel for reduction(+:count)
    for (int i = 0; i < N; i++) {
        if (arr[i] == key) {
            count++;
        }
    }
    printf("Total occurrences of key %d: %d\n", key, count);
    return 0;
}
