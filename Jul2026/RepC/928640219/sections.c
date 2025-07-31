#include <omp.h>
#include <stdio.h>

void f1()
{
    printf("Thread %d executando tarefa A\n", omp_get_thread_num());
}

void f2()
{
    printf("Thread %d executando tarefa B\n", omp_get_thread_num());
}

int main(){
    #pragma omp parallel
    {
        #pragma omp sections
        {
            #pragma omp section
            {
                f1();
            }

            #pragma omp section
            {
                f2();
            }
        }
    }
    
    return 0;
}
