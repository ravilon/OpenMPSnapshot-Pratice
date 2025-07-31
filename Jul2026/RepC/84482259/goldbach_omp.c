/*
Calculations for the Goldbach conjecture - OpenMP
*/

#include<stdio.h>
#include<stdlib.h>
#include"timer.h"
#include<omp.h>

#define N 10000
#define NUM_THREADS 8
#define CHUNK 1000

//check if prime number
int isprime(int m){
	int i;
	for(i=2;i<=m/2;i++){
		if(m%i==0){
		  return 0;
		}
	}
	return 1;
}

//main function
int main(){
	int i,j,r1,r2,tid;
	int **isgold=NULL;

	FILE *fgbach2_omp = fopen("fgbach2_omp.dat","w+");

	timespec before,after;
	get_time(&before);

	//goldbach condition check
	//check if even
	if (N%2 != 0){
	  return 0;
	}
	//check if less than 2
	if (N<=2){
	  return 0;
	}

	//allocate memory to isgold
	isgold = (int**)malloc((N/2+1)*sizeof(int*));
	for(i=0;i<=N/2;i++){
	  isgold[i] = (int*)malloc((N/2+1)*sizeof(int));
	}
	for(i=0;i<=N/2;i++){
	  for(j=0;j<=N/2;j++){
	    isgold[i][j]=0;
	  }
	}

	//goldbach
	omp_set_num_threads(NUM_THREADS);
	#pragma omp parallel shared(isgold) private(i,j,r1,r2)
	{
		tid = omp_get_thread_num();
		printf("%d\n",tid);
		#pragma omp for schedule(guided)
		for(j=4;j<=N;j+=2){

			for(i=3;i<=j/2;i++){
				r1 = isprime(i);
				r2 = isprime(j-i);
				if(r1==1 && r2==1){
					isgold[i][j/2]=1;
				}
			}
		}
	}

	get_time(&after);

	timespec time_diff;
	diff(&before,&after,&time_diff);

	//time in sec
	double time_s = time_diff.tv_sec + (double)(time_diff.tv_nsec)/1.0e9;
	printf("time = %.09lf \n", time_s);

	//print to file
	for(i=0;i<=N/2;i++){
	  for(j=0;j<=N/2;j++){
	    if(isgold[i][j]==1){
	      fprintf(fgbach2_omp,"%d\t%d\t%d\n", j*2, i, j*2-i);
	    }
	  }
	}
	fclose(fgbach2_omp);
	return 0;
}




