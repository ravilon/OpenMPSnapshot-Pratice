#include <iostream>
#include <omp.h>

bool compare_results(int* arr1, int* arr2, int n, int x1, int x2) {
	bool match = x1 == x2;
	if (match) {
		for (int i = 0; i < n; i++) {
			if (arr1[i] != arr2[i]) {
				match = false;
				break;
			}
		}
	}
	return match;
}

int main()
{
	std::cout << std::fixed;

	int num_threads = omp_get_num_procs();
	omp_set_num_threads(num_threads);

	constexpr int n = 200, m = 100;
	int x_seq, x_par;
	int* a_seq = new int[n * m];
	int* a_par = new int[n * m];

	for (int i = 0; i < n * m; i++) {
		a_seq[i] = a_par[i] = i + 1;
	}

	double start, end;
	start = omp_get_wtime();

	x_seq = 0;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			x_seq += a_seq[i * m + j];
			a_seq[i * m + j] *= 2;
		}
	}

	end = omp_get_wtime();
	std::cout << "Serial: " << end - start << "s\n";

	start = omp_get_wtime();

	x_par = 0;
	#pragma omp parallel for reduction(+:x_par)
	for (int i = 0; i < n * m; i++) {
		int row = i / m, col = i % m;
		x_par += a_par[row * m + col];
		a_par[row * m + col] *= 2;
	}

	end = omp_get_wtime();
	std::cout << "Parallel: " << end - start << "s\n";

	std::cout << (compare_results(a_seq, a_par, n, x_seq, x_par) ? "CORRECT" : "INCORRECT") << "\n";

	delete[] a_seq;
	delete[] a_par;
}