#include<stdio.h>
#include<omp.h>

void teste(){
    int i, x[10] = {1,2,3,4,5,6,7,8,9,10}, N=10, maxval = 5;

    #pragma omp parallel for
    for (i = 0; i < N ; i ++)
        if (x [ i ] > maxval) 
            break; //break não pode ser utilizado com parallel for
}

int main(int argc, char *argv[]){
    
    omp_set_num_threads(4);

    teste();

    return 0;
}