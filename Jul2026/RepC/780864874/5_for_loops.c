#include<stdio.h>
#include<omp.h>


int main()
{
   double start_time=omp_get_wtime();
   printf("Start Time: %f\n",start_time);

   double tick=omp_get_wtick();
   double elapsed_time=0;
   int N=10;
   int sum=0;

   #pragma omp parallel for

    for(int i=1;i<=N;i++){
        printf("%d Thread is processing iteration %d\n",omp_get_thread_num(),i);
        sum+=i;
    }

    printf("Sum is %d\n",sum);


   double end_time=omp_get_wtime();
   printf("End Time: %f\n",end_time);
   elapsed_time=end_time-start_time;
    printf("Elapsed Time: %f\n",elapsed_time);
   printf("Time Resolution: %f\n",tick);

}

