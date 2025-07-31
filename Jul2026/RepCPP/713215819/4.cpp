#include<omp.h>
#include<stdio.h>

int main(){
 
	#pragma omp parallel
	{
		printf("Number of threads are %d\n", omp_get_num_threads());
		printf("Hello i am thread %d\n", omp_get_thread_num()) ;
	}
	
	return 0;
}
