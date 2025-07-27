#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


int main(int argc, char *argv[]){
    int a = 10;
    omp_set_num_threads(4);
    
    #pragma omp parallel /*first*/private(a) /*Variáveis privadas não possuem valor inicial, para dar um valor inicial é necessário utilizar firstprivate*/
    {
        printf("%d\n", a);
    }

    
    return 0;
}