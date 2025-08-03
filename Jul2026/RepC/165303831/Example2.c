//#include <stdio.h>
//#include <omp.h>
//#include <conio.h>
//
//int main(int argc, char *argv[])
//{
//	int tid, tsize;
//#pragma omp parallel num_threads(5)
//	{
//		tsize = omp_get_num_threads();
//		printf("Number of Threads = %d \n", tsize);
//#pragma omp sections
//		{
//#pragma omp section 
//			{
//				tid = omp_get_thread_num();
//				printf("1st Thread %d is done. \n", tid); 
//			}
//#pragma omp section
//			{
//				tid = omp_get_thread_num();
//				printf("2nd Thread %d is done. \n", tid); 
//			}
//#pragma omp section
//			{
//				tid = omp_get_thread_num();
//				printf("3rd Thread %d is done. \n", tid); 
//			}
//#pragma omp section
//			{
//				tid = omp_get_thread_num();
//				printf("4th Thread %d is done. \n", tid);
//			}
//#pragma omp section
//			{
//				tid = omp_get_thread_num();
//				printf("5th Thread %d is done. \n", tid); 
//			}
//		} /*--The end of the sections block--*/
//	}
//	_getch(); // for keep console from <conio.h> library
//	return 0;
//}