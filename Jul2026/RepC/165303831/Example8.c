//#include <stdio.h>
//#include <omp.h>
//#include <conio.h>
//
//int main(int argc, char *argv[])
//{
//	int i, ic = 0, n = 4;
//#pragma omp parallel num_threads(3) shared(n,ic) private (i) 
//	{
//#pragma omp for  
//		for (i = 0; i < n; i++)
//		{
//			printf("%d,iteration is carried out by thread (%d). \n", i, omp_get_thread_num());
//#pragma omp atomic
//			ic += 1;
//		}
//	}
//
//	printf("Count = %d\n", ic);
//	_getch(); // for keep console from <conio.h> library
//	return 0;
//}
//
