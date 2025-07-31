#include <omp.h>
#include <iostream>

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
	int b[n], c[n], t_par, t_seq, a_par[n], a_seq[n], a_copy[n - 1];

	for (int i = 0; i < n; i++) {
		b[i] = c[i] = a_par[i] = a_seq[i] = i + 1;
	}

	double start, end;
	start = omp_get_wtime();

	t_seq = 1;
	for (int i = 0; i < n - 1; i++) {
		a_seq[i] = a_seq[i + 1] + b[i] * c[i];
		t_seq *= a_seq[i];
	}

	end = omp_get_wtime();
	std::cout << "Serial: " << end - start << "s\n";

	start = omp_get_wtime();

	#pragma omp parallel for
	for (int i = 0; i < n - 1; i++) {
		a_copy[i] = a_par[i + 1];
	}

	t_par = 1;
	#pragma omp parallel for reduction(*:t_par)
	for (int i = 0; i < n - 1; i++) {
		a_par[i] = a_copy[i] + b[i] * c[i];
		t_par *= a_par[i];
	}

	end = omp_get_wtime();
	std::cout << "Parallel: " << end - start << "s\n";

	std::cout << (compare_results(a_seq, a_par, n, t_seq, t_par) ? "CORRECT" : "INCORRECT") << "\n";
}