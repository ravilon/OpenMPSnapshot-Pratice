/*
*	AUTHOR: Pratik Kataria
*/

#include<stdio.h>
#include<stdlib.h>
#include<omp.h>

int arr[65536],numSearch, flag, n = 8, global_size, global_x;
double start_time;

//PROTOTYPE
void arrayCreation(int);
void nArySearch(int, int);

void main(){
	int x = 0;
	arrayCreation(65536);
	printf("Enter number to be searched \n");
	scanf("%d", &numSearch);
	// Start timer
	start_time = omp_get_wtime();
	nArySearch(65536, x);
	printf("\nExecution time = %lf seconds\n", omp_get_wtime() - start_time);
}

void arrayCreation(int size){

	for(int i = 0 ; i < size ; i++)
		arr[i] = i*2;		

}

void nArySearch(int size, int x){
	if(size <= 8){
		//We're down to last 8 elements so go into this.
		flag=0;	
		#pragma omp parallel
		{
			//Each thread in parallel has 1 element of the last 8 elements
			int tid=omp_get_thread_num();
			//global_x denotes starting position of the array block
			if(arr[global_x+tid]==numSearch)
			{

				printf("Number found at %d\n",global_x+tid);
				flag=1;
			}
		}
		
		if(flag==0)
		{

			printf("Number not found\n");
		}
		
	
	}
	else{
		//FIRST IT GOES HERE
		flag = 0;
		#pragma omp parallel
		{
			int tid = omp_get_thread_num();
/*

---------ALL COMMENTED CALCULATION FOR THREAD ID 0 & 1st ITERATION ONLY-----------

arr[0*(65536/8)+0] = arr[0*(8192)+0] = arr[0]	
arr[0*(65536/8)+0+(65536/8)-1] = arr[0*(8192)+0+(8191)] = arr[8191]
SIMILARLY YOU CAN UNDERSTAND REST OF THE CALCULATION
*/
			printf("Range: %d -- %d \tThread ID: %d \t CPU: %d\n",arr[tid*(size/n)+x], arr[tid*(size/n)+x+(size/n)-1], tid, sched_getcpu());
			//if number to be searched is greater or equal than lower bound and lower or equal than higher bound
			if(numSearch >= arr[tid*(size/n)+x] && numSearch <= arr[tid*(size/n)+x+(size/n)-1])
			{
				printf("Looking in range  %d -- %d which has size=(   %d   ) and global_x = %d\n",arr[tid*size/n+x],arr[tid*size/n+size/n-1+x],+size/n, global_x);
				//global_x gives starting location of array block which would be finally searched and x = global_x
				global_size=size/n;				//global_size = 8192					
				global_x=tid*global_size+x;		//global_x = 0
				flag=1;
			}
		}
		if(flag == 1)
			//Number will be found so flag = 1
			nArySearch(global_size,global_x);
		else
			printf("Not Found");		
	
	}

}

/* FOR CPU WITH 8 CORES (Intel i7), n = 8

Enter number to be searched 
24
Range: 65536 -- 81918 	Thread ID: 4 	 CPU: 7
Range: 0 -- 16382 	Thread ID: 0 	 CPU: 6
Looking in range  0 -- 16382 which has size=(   8192   ) and global_x = 0
Range: 16384 -- 32766 	Thread ID: 1 	 CPU: 0
Range: 114688 -- 131070 	Thread ID: 7 	 CPU: 1
Range: 98304 -- 114686 	Thread ID: 6 	 CPU: 2
Range: 32768 -- 49150 	Thread ID: 2 	 CPU: 3
Range: 49152 -- 65534 	Thread ID: 3 	 CPU: 5
Range: 81920 -- 98302 	Thread ID: 5 	 CPU: 4
Range: 0 -- 2046 	Thread ID: 0 	 CPU: 6
Looking in range  0 -- 2046 which has size=(   1024   ) and global_x = 0
Range: 12288 -- 14334 	Thread ID: 6 	 CPU: 4
Range: 14336 -- 16382 	Thread ID: 7 	 CPU: 1
Range: 2048 -- 4094 	Thread ID: 1 	 CPU: 0
Range: 6144 -- 8190 	Thread ID: 3 	 CPU: 5
Range: 4096 -- 6142 	Thread ID: 2 	 CPU: 2
Range: 10240 -- 12286 	Thread ID: 5 	 CPU: 3
Range: 8192 -- 10238 	Thread ID: 4 	 CPU: 7
Range: 1536 -- 1790 	Thread ID: 6 	 CPU: 4
Range: 768 -- 1022 	Thread ID: 3 	 CPU: 5
Range: 0 -- 254 	Thread ID: 0 	 CPU: 6
Looking in range  0 -- 254 which has size=(   128   ) and global_x = 0
Range: 1024 -- 1278 	Thread ID: 4 	 CPU: 7
Range: 1792 -- 2046 	Thread ID: 7 	 CPU: 1
Range: 512 -- 766 	Thread ID: 2 	 CPU: 2
Range: 256 -- 510 	Thread ID: 1 	 CPU: 0
Range: 1280 -- 1534 	Thread ID: 5 	 CPU: 3
Range: 160 -- 190 	Thread ID: 5 	 CPU: 3
Range: 64 -- 94 	Thread ID: 2 	 CPU: 2
Range: 224 -- 254 	Thread ID: 7 	 CPU: 1
Range: 96 -- 126 	Thread ID: 3 	 CPU: 5
Range: 192 -- 222 	Thread ID: 6 	 CPU: 4
Range: 32 -- 62 	Thread ID: 1 	 CPU: 0
Range: 0 -- 30 	Thread ID: 0 	 CPU: 7
Looking in range  0 -- 30 which has size=(   16   ) and global_x = 0
Range: 128 -- 158 	Thread ID: 4 	 CPU: 6
Range: 8 -- 10 	Thread ID: 2 	 CPU: 2
Range: 0 -- 2 	Thread ID: 0 	 CPU: 7
Range: 16 -- 18 	Thread ID: 4 	 CPU: 6
Range: 4 -- 6 	Thread ID: 1 	 CPU: 0
Range: 20 -- 22 	Thread ID: 5 	 CPU: 3
Range: 28 -- 30 	Thread ID: 7 	 CPU: 1
Range: 12 -- 14 	Thread ID: 3 	 CPU: 5
Range: 24 -- 26 	Thread ID: 6 	 CPU: 4
Looking in range  24 -- 26 which has size=(   2   ) and global_x = 0
Number found at 12

Execution time = 0.025089 seconds
kpratik@pratik-lenovo:~/Downloads/PL3$ 
*/
