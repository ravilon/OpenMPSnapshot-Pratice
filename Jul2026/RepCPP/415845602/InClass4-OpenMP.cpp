#include "pch.h"
#include <iostream>
#include <cstdio>
#include <omp.h>

using namespace std;

int main()
{
	int a = 3;
	int b[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	int c[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	int d[10];

	cout << "1 - " << a << endl;

	omp_set_num_threads(3);

	//#pragma omp parallel shared(a) num_threads(3)
	#pragma omp parallel private(a) shared(b, c, d)
	{
		//a = 10;
		//cout << "2 - " << a << endl;
		//printf("2 - %d-%d\n", a, omp_get_thread_num());
		#pragma omp for schedule(dynamic, 2) nowait
			for (int i = 0; i < 10; i++) {
				d[i] = b[i] + c[i];
				printf("%d-%d\n", i, omp_get_thread_num());
			}
		
			cout << "fin for\n";
	} // end parallel

	/*#pragma omp sections
	{
		#pragma omp section
		{
			printf("2 - %d-%d\n", a + 1, omp_get_thread_num());
		}
		#pragma omp section
		{
			printf("2 - %d-%d\n", a + 2, omp_get_thread_num());
		}
		#pragma omp section
		{
			printf("2 - %d-%d\n", a + 3, omp_get_thread_num());
		}
	}*/

	/*#pragma omp critical
	{

	}*/

	/*#pragma omp master
	{

	}*/

	//cout << "3 - " << a << endl;

	return 0;
}
