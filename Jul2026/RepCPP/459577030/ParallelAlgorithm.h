#pragma once
#include "SerialAlgorithm.h"
#include <omp.h>


matrix parallelCalcDiscrepancy(matrix& r, const matrix& V, double param_x, double param_y, double A, int a, int b, int c, int d, size_t n, size_t m, size_t threads) {
	double h = 2 / (double)n;
	double k = 2 / (double)n;
	int i, j;
	omp_set_num_threads(threads);
#pragma omp parallel for private (j)
	for ( i = 1; i < m; i++) {
		for ( j = 1; j < n; j++) {
			r[i - 1][j - 1] = V[i][j] * A + (V[i][j + 1] + V[i][j - 1]) / param_x + (V[i + 1][j] + V[i - 1][j]) / param_y + Laplas(a + j * h, c + i * k);
		}
	}
	return r;
}


matrix parallelCalcAh(matrix& H, matrix& r, double A, double param_x, double param_y, size_t n, size_t m, size_t threads) {
	matrix Ah;
	Ah.assign(m, vector<double>(n));
	int i, j;
	omp_set_num_threads(threads);
#pragma omp parallel for private (j)
	for (i = 1; i < r.size(); i++) {
		for (j = 1; j < r[0].size(); j++) {
			Ah[i - 1][j - 1] = A * H[i - 1][j - 1] + (H[i - 1][j] + H[i - 1][j]) / param_x + (H[i][j - 1] + H[i][j - 1]) / param_y; // (A,h)			
		}
	}
	return Ah;
}


double parallelCalcAhh(matrix& Ah, matrix& r, size_t threads) {
	double temp = 0;
	int i, j;
	omp_set_num_threads(threads);
#pragma omp parallel for shared(Ah, r) private(j) reduction(+:temp) 
	for (i = 1; i < r.size(); i++) {
		for (j = 1; j < r[0].size(); j++) {
			temp += Ah[i - 1][j - 1] * r[i - 1][j - 1];// (Ah,h)
		}
	}
	return temp;
}

matrix parallelCalcH(matrix& H, matrix& r, double betta, size_t threads) {
	int i, j;
	omp_set_num_threads(threads);
#pragma omp parallel for private (j)
	for (i = 1; i < H.size(); i++) {
		for (j = 1; j < H[0].size(); j++) {
			H[i - 1][j - 1] = r[i - 1][j - 1] * (-1) + H[i - 1][j - 1] * betta; // h^(s) = betta * h^(s-1) - r^(s)
		}
	}
	return  H;
}


double parallelCalcAlpha(matrix& H, matrix& r, double param_x, double param_y, double A, double& Ahh, size_t n, size_t m, size_t threads) {
	int asdthread = threads;
	double temp = 0, betta = 0, alpha = 0;
	matrix Ah = parallelCalcAh(H, r, A, param_x, param_y, n, m, threads);
	temp = parallelCalcAhh(Ah, r, threads);
	//	temp = calcAhh(Ah, r); 
	betta = temp / Ahh;
	H = parallelCalcH(H, r, betta, threads);
	//	H = calcH(H, r, betta);
	Ahh = 0; temp = 0;
	Ah = parallelCalcAh(H, r, A, param_x, param_y, n, m, threads);
	//	Ah = calcAh(H, r, A, param_x, param_y, n, m);
	Ahh = parallelCalcAhh(Ah, H, threads);
	//	Ahh = calcAhh(Ah, H);
	temp = parallelCalcAhh(r, H, threads);
	//	temp = calcAhh(r, H);
	alpha = -temp / Ahh;

	return alpha;

}