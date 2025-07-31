#include<omp.h>
#include<stdio.h>

int main(){
	char name[] = "Dhivyesh RK", roll[] = "2021BCS0084";
 
	#pragma omp parallel
	{
		printf("Name is %s and roll is %s. Thread id is %d\n", name, roll, omp_get_thread_num());
	}
	
	return 0;
}
