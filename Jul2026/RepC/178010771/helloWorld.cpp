#include <stdio.h>
#include <omp.h>

int main()
{
   #pragma omp parallel
   {
        int ID = omp_get_thread_num(); //retorna o identificador da thread
    	printf("Hello(%d) ",ID); //Queremos escrever o id de cada thread
    	printf("World(%d)\n",ID);
   }

	return 0;
}