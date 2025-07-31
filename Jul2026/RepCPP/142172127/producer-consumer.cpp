
#include "bits/stdc++.h"
#include "omp.h"

#define N 10000
#define Nthreads 2

/* Some random number constants from numerical recipies */
#define SEED       2531
#define RAND_MULT  1366
#define RAND_ADD   150889
#define RAND_MOD   714025
int randy = SEED;

void fill_rand(int length, double *a)
{
    int i; 
    for (i=0;i<length;i++)                                    //fill buffer with random numbers
    {
        randy = (RAND_MULT * randy + RAND_ADD) % RAND_MOD;
        *(a+i) = ((double) randy)/((double) RAND_MOD);
    }   
}

double Sum_array(int length, double *a)
{
    int i;  double sum = 0.0;
    for (i=0;i<length;i++)                                    //Consumption by consumer
    {
        sum += *(a+i);
    }
    return sum; 
}
  
int main()
{
    double *A, sum, runtime;
    int numthreads, flag = 0,flag_tmp;

    omp_set_num_threads(Nthreads);

    A = (double *)malloc(N*sizeof(double));

    #pragma omp parallel
    {
        #pragma omp master
        {
            numthreads = omp_get_num_threads();
            if(numthreads != 2)
            {
                printf("error: incorect number of threads, %d \n",numthreads);
                exit(-1);
            }
            runtime = omp_get_wtime();
        }
        #pragma omp barrier                                                      //setting 2 threads (1 producer 1 consumer)
        
        #pragma omp sections
        {
            #pragma omp section                                                 //producer section
            { 
                fill_rand(N, A);                                                //fill buffer
                #pragma omp flush                                               //flush to make it visible to all threads
                #pragma atomic write                                            //atomic write to write 1 completely
                    flag = 1;
                #pragma omp flush (flag)                                        // flush to make flag value visible meaning telling consumer I have produced
            }                                                                   //you can consume
            #pragma omp section                                                 //conusmer section
            {
                while(1)
                {
                  #pragma omp flush(flag)                                       //Looks at flag value
                  #pragma omp atomic read                                       //reads it completely
                     flag_tmp = flag;                 
                  if(flag_tmp == 1)                                             //if producer has written then consumes
                    break;
                }
                  #pragma omp flush                                             //makes sum visible to all
                    sum = Sum_array(N, A);
            }
        }
        #pragma omp master
            runtime = omp_get_wtime() - runtime;
    }  

    printf("With %d threads and %lf seconds, The sum is %lf \n",numthreads,runtime,sum);
}
