//#include <stdio.h>
//#include <omp.h>
//#include <conio.h>
//
//int main(int argc, char *argv[])
//{
//	int tid,tsize;
//	tid = omp_get_thread_num();
//	printf("Thread Number = %d \n", tid);
//	tsize = omp_get_num_threads();
//	printf("Number of Threads = %d \n", tsize);
//	omp_set_num_threads(8);
//
//#pragma omp parallel 
//	{
//		tid = omp_get_thread_num();
//		tsize = omp_get_num_threads();
//
//		printf("Hello Thread = %d / %d\n", tid, tsize);
//	}
//	tid = omp_get_thread_num();
//	printf("Thread %d is done. \n", tid);
//
//	_getch(); // for keep console from <conio.h> library
//	return 0;
//}