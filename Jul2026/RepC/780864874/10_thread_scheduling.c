#include<stdio.h>
#include<omp.h>

int main()
{
    const int N=10;
    const int chunk_size=2;
    int chunk=0;
    omp_sched_t kind;

    omp_set_schedule(omp_sched_static,chunk_size);
    omp_get_schedule(&kind,&chunk);

    #pragma omp parallel for schedule(runtime)
    for(int i=0;i<N;i++)
    {
        printf("This is %d from Thread %d\n",i,omp_get_thread_num());
    }

    printf("This loop works on policy : %d\n",kind);
    printf("Chunk size : %d\n",chunk);
}

/*
    Following method of defining the policy is correct.
    If do so no need to use omp_set_schedule() method.

    #pragma omp parallel for schedule(static, chunk_size)
    for(int i=0;i<N;i++)
    {
        printf("This is %d from Thread %d\n",i,omp_get_thread_num());
    }
*/
