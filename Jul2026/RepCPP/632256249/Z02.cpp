#include <iostream>
#include <omp.h>
#include <vector>

void init_vec(std::vector<int>& v) {
	for (int i = 0; i < v.size(); i++) {
		v[i] = i + 1;
	}
}

int main()
{
	int num_threads = omp_get_num_procs();
	omp_set_num_threads(num_threads);
	std::cout << "Number of threads: " << num_threads << "\n\n";

	std::cout << std::fixed;

	std::vector<int> dims{ 10, 50, 100, 500, 1500, 5000 };
	for (const int& n : dims) {
		std::vector<int> a(n), b(n);
		init_vec(a);
		init_vec(b);

		int result_serial = 0, result_parallel = 0;
		double start, end;

		std::cout << "N: " << n << "\n";

		start = omp_get_wtime();
		{
			for (int i = 0; i < n; i++) result_serial += a[i] * b[i];
		}
		end = omp_get_wtime();
		std::cout << "Serial: " << end - start << "s\n";

		start = omp_get_wtime();
		{
			#pragma omp parallel for
			for (int i = 0; i < n; i++) {
				#pragma omp critical
				result_parallel += a[i] * b[i];
			}
		}
		end = omp_get_wtime();
		std::cout << "Parallel: " << end - start << "s\n";

		result_parallel = 0;
		start = omp_get_wtime();
		{
			#pragma omp parallel for reduction(+:result_parallel)
			for (int i = 0; i < n; i++) {
				result_parallel += a[i] * b[i];
			}
		}
		end = omp_get_wtime();
		std::cout << "Parallel with reduction: " << end - start << "s\n\n";
	}
}