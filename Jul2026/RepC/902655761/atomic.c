#include <stdio.h>
#include <omp.h>

#define n 10
int main(){
    int ic, i;

    ic = 0;

    #pragma omp parallel shared(ic) private(i)
    for(i=0;i<n;i++){
            #pragma omp atomic
            ic++;
    }

    printf("counter = %d\n", ic);

    return 0;
}