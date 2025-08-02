//#include <stdio.h>
//#include <omp.h>
//#include <conio.h>
//
//int main(int argc, char *argv[])
//{
//	int tid, n = 6;
//#pragma omp parallel if (n > 5) private(tid) shared(n) num_threads(3) // It's work as Parallel
//	{
//		tid = omp_get_thread_num();
//#pragma omp single
//		{
//			printf("Value of n = %d\n", n);
//			printf("Size of Threads = %d\n", omp_get_num_threads());
//		}
//		printf("Print statement executed by Thread (%d) \n", tid);
//	} /*-- End to parallel segment --*/
//
//
//	printf("--------------------------------------------------\n\n");
//
//#pragma omp parallel if (n > 8) private(tid) shared(n) num_threads(3) // It's work as Serial (as main)
//	{
//		tid = omp_get_thread_num();
//#pragma omp single
//		{
//			printf("Value of n = %d\n", n);
//			printf("Size of Threads = %d\n", omp_get_num_threads());
//		}
//		printf("Print statement executed by Thread (%d) \n", tid);
//	} /*-- End to parallel segment --*/
//
//
//	_getch(); // for keep console from <conio.h> library
//	return 0;
//}