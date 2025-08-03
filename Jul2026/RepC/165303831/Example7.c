//#include <stdio.h>
//#include <omp.h>
//#include <conio.h>
//
//int main(int argc, char *argv[])
//{
//	int tid, totalLocal=0, total=0, n = 6, i, a[] = {1,2,3,4,5,6};
//#pragma omp parallel num_threads(3) shared(n,a,total) private(tid, totalLocal) 
//	{
//		tid = omp_get_thread_num();
//		totalLocal = 0;
//#pragma omp for
//		for (i = 0; i < n; i++)
//			totalLocal += a[i];
//#pragma omp critical (total)
//		{
//			total += totalLocal;
//			printf("tid = %d : totalLocal =%d total = %d\n", tid, totalLocal, total);
//		}
//	} /*-- End of for loop --*/
//
//	printf("The value of the total after the parallel region: %d\n", total);
//
//	_getch(); // for keep console from <conio.h> library
//	return 0;
//}