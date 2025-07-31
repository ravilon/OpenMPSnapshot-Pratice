#include<omp.h>
#include<stdio.h>

int main(){
	char name[] = "Dhivyesh RK", roll[] = "2021BCS0084";
    omp_set_num_threads(2);
	#pragma omp parallel
	{
		printf("Name is %s and roll is %s\n", name, roll);
	}
	
	return 0;
}
