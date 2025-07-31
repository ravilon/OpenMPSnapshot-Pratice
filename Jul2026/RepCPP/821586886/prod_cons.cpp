#include <omp.h>
#include <iostream>

#define N 1000000

// Some random number constants
#define SEED       2531
#define RAND_MULT  1366
#define RAND_ADD   150889
#define RAND_MOD   714025

int randy = SEED;


// function to fill an array with random numbers
void fill_rand(const int& length, double* a){
    for (int i=0;i<length;i++){
        randy = (RAND_MULT * randy + RAND_ADD) % RAND_MOD;
        *(a+i) = ((double) randy)/((double) RAND_MOD);
    }
}


// function to sum the elements of an array
double Sum_array(const int& length, double* a){
    double sum=0.0;

    for (int i=0;i<length;i++){
        sum += *(a+i);
    }

    return sum; 
} 


int main(){
    double sum, start_time, run_time;

    int flag = 0;
    int flag_mp =0;

    double* A = new double[N];

    int n_threads;

    std::cout<<"Enter Number Of Threads:";
    std::cin>>n_threads;
    omp_set_num_threads(n_threads);

    start_time = omp_get_wtime();

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            fill_rand(N, A); // fill an array of data
            #pragma omp flush

            #pragma omp atomic write
            flag = 1;
            #pragma omp flush(flag)
        }

        #pragma omp section
        {
            while(1){
                #pragma omp flush(flag)

                #pragma omp atomic read
                flag_mp = flag;

                if(flag_mp == 1) {break;}
            }

            #pragma omp flush
            sum = Sum_array(N, A); // sum the array
        }
    }

    run_time = omp_get_wtime() - start_time;

    std::cout <<" In "<<run_time<<" seconds, The sum is "<<sum<<std::endl;

    delete[] A;
    return 0;
}