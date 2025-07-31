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

	constexpr int n = 200;
	int x_seq, x_par, res[n], add[n], sum_seq[n], sum_par[n], sum_copy[n - 2];

	for (int i = 0; i < n; i++) {
		res[i] = add[i] = sum_seq[i] = sum_par[i] = i + 1;
	}

	double start, end;
	start = omp_get_wtime();

	for (int i = n - 1; i > 1; i--) {
		x_seq = res[i] + add[i];
		sum_seq[i] = sum_seq[i - 1] + x_seq;
	}

	end = omp_get_wtime();
	std::cout << "Serial: " << end - start << "s\n";

	start = omp_get_wtime();

	#pragma omp parallel for 
	for (int i = 0; i < n - 2; i++) {
		sum_copy[i] = sum_par[i + 1];
	}

	#pragma omp parallel for lastprivate(x_par)
	for (int i = n - 1; i > 1; i--) {
		x_par = res[i] + add[i];
		sum_par[i] = sum_copy[i - 2] + x_par;
	}

	end = omp_get_wtime();
	std::cout << "Parallel: " << end - start << "s\n";

	std::cout << (compare_results(sum_seq, sum_par, n, x_seq, x_par) ? "CORRECT" : "INCORRECT") << "\n";
}