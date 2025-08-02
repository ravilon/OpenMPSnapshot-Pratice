//#include <stdio.h>
//#include <omp.h>
//#include <conio.h>
//
//int main(int argc, char *argv[])
//{
//	int i, n = 6;
//#pragma omp parallel num_threads(6) shared(n) private (i) 
//	{
//#pragma omp for nowait schedule(dynamic) // Not Fair
//		for (i = 0; i < n; i++)
//		{
//			printf(" Thread (%d) executes loop repetition with %d \n",omp_get_thread_num(), i);
//
//		}
//		printf("No wait \n");
//
//	}
//
//	printf("--------------------------------------------------\n\n");
//
//#pragma omp parallel num_threads(6) shared(n) private (i) 
//	{
//#pragma omp for nowait schedule(static) // Fair
//		for (i = 0; i < n; i++)
//		{
//			printf(" Thread (%d) executes loop repetition with %d \n", omp_get_thread_num(), i);
//
//		}
//		printf("No wait \n");
//
//	}
//
//	_getch(); // for keep console from <conio.h> library
//	return 0;
//}