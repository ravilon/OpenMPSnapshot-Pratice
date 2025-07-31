//#include <stdio.h>
//#include <omp.h>
//#include <conio.h>
//
//int main(int argc, char *argv[])
//{
//	int tid;
//#pragma omp parallel num_threads(4) shared(tid)
//	{
//#pragma omp single //Single structure is executed by any thread
//		{
//			tid = omp_get_thread_num();
//			printf("Single structure is executed by thread %d \n", tid);
//		}
//		/* A barrier is automatically added here */
//		printf("This code is executed by the thread %d.\n", omp_get_thread_num());
//	}
//	_getch(); // for keep console from <conio.h> library
//	return 0;
//}