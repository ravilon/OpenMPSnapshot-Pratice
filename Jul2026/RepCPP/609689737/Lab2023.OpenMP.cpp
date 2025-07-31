#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

/*
4. Napisati sekvencijalni program kojim se generiše skalarni proizvod dva vektora. Koriščenjem
OpenMP direktive izvršiti paralelizaciju, podelom iteracija petlje nitima, sa i bez korišćenja
odredbe redukcije za kombinovanje parcijalnih rezultata u nitima. Uporediti vremena izvršenja u
oba slučaja sa sekvencijalnim vremenom izvršenja. Uporediti ova rešenja za različite dimenzije
vektora. Testirati za različit broj niti i različitu podelu iteracija između niti.
*/

void dot_product(int N, int num_threads) {
	omp_set_num_threads(num_threads);
	printf("num_threads : %d\n", num_threads);

	int* A = (int*)malloc(sizeof(int) * N);
	int* B = (int*)malloc(sizeof(int) * N);
	int rez = 0;
	double time = 0, t1, t2, t3;
	for (int i = 0; i < N; i++) {
		A[i] = i;
		B[i] = N - i;
	}
	printf("N = %d:s\n", N);
	time = omp_get_wtime();
	for (int i = 0; i < N; i++) {
		rez += A[i] * B[i];
	}
	t1 = omp_get_wtime() - time;
	time = omp_get_wtime();
	rez = 0;
#pragma omp parallel for
	for (int i = 0; i < N; i++) {
		int rez2 = A[i] * B[i];
#pragma omp critical
		rez += rez2;
	}
	t2 = omp_get_wtime() - time;
	rez = 0;
	time = omp_get_wtime();
#pragma omp parallel for reduction(+: rez)
	for (int i = 0; i < N; i++)
		rez += A[i] * B[i];
	t3 = omp_get_wtime() - time;

	printf("Sequential time :\t %4.5lfms\n", t1 * 1000);
	printf("Critical time :\t %4.5lfms, speedup : %2.2lf\n", t2 * 1000, t1 / t2);
	printf("Reduction time :\t %4.5lfms, speedup : %2.2lf\n\n", t3 * 1000, t1 / t3);

	free(A);
	free(B);
}

int main() {

	dot_product(5, 2);
	dot_product(10, 2);
	dot_product(40, 2);
	dot_product(100, 2);
	dot_product(1000, 2);
	dot_product(50000, 2);
	dot_product(100000, 2);
	dot_product(5000000, 2);

	dot_product(5, 8);
	dot_product(10, 8);
	dot_product(40, 8);
	dot_product(100, 8);
	dot_product(1000, 8);
	dot_product(50000, 8);
	dot_product(100000, 8);
	dot_product(5000000, 8);

	return 0;
}
