//#include <stdio.h>
//#include <omp.h>
//#include <conio.h>
//
//int main(int argc, char *argv[])
//{
//	int i, s = 0, n = 6;
//#pragma omp parallel num_threads(6)  private(i) firstprivate(s)
//	{
//#pragma omp for
//		for (i = 1; i < n; i++)
//		{
//			s = i + 1;
//			printf("For %d thread s = %d AND i = %d\n",omp_get_thread_num(), s, i);
//		}
//
//		printf("After the parallel loop, value of a = %d \n", s);
//	}
//	
//	_getch(); // for keep console from <conio.h> library
//	return 0;
//}