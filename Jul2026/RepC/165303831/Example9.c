//#include <stdio.h>
//#include <omp.h>
//#include <conio.h>
//
//int main(int argc, char *argv[])
//{
//	int i, a, n = 6, b[] = { 0,0,0,0,0,0 };
//#pragma omp parallel shared(a,b) private(i) num_threads(3)
//	{
//#pragma omp master
//		{
//			a = 10;
//			printf("Master structure is executed by thread (%d) \n",
//				omp_get_thread_num());
//		}
//#pragma omp barrier //There must be a barrier here.
//#pragma omp for
//		for (i = 0; i < n; i++)
//		{
//			b[i] = a;
//			printf("%d, Iteration carried out by thread %d :\n", i, omp_get_thread_num());
//		}
//	}
//	_getch(); // for keep console from <conio.h> library
//	return 0;
//}