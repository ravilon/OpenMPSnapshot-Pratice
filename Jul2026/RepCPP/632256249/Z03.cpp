#include <iostream>
#include <omp.h>
#include <vector>

bool is_prime(unsigned int n) {
	if (n == 2 || n == 3) return true;
	if (n <= 1 || n % 2 == 0 || n % 3 == 0) return false;

	for (int i = 5; i * i <= n; i += 6) {
		if (n % i == 0 || n % (i + 2) == 0) return false;
	}

	return true;
}

int main()
{
	int num_threads = omp_get_num_procs();
	omp_set_num_threads(num_threads);
	std::cout << "Number of threads: " << num_threads << "\n\n";

	std::cout << std::fixed;

	std::vector<int> dims{ 10, 50, 100, 500, 1500, 5000, 20000, 500000 };
	for (const int& n : dims) {
		int num_serial = 0, num_parallel = 0;

		double start, end;

		std::cout << "N: " << n << "\n";

		start = omp_get_wtime();
		{
			for (int i = 1; i < n; i++) {
				if (is_prime(i)) num_serial++;
			}
		}
		end = omp_get_wtime();
		std::cout << "Serial: " << end - start << "s\n";

		start = omp_get_wtime();
		{
			#pragma omp parallel for schedule(dynamic, 5)
			for (int i = 1; i < n; i++) {
				if (is_prime(i)) {
					#pragma omp critical
					num_parallel++;
				}
			}
		}
		end = omp_get_wtime();
		std::cout << "Parallel: " << end - start << "s\n";

		num_parallel = 0;
		start = omp_get_wtime();
		{
			#pragma omp parallel for reduction(+:num_parallel) schedule(dynamic, 5)
			for (int i = 1; i < n; i++) {
				if (is_prime(i)) num_parallel++;
			}

		}
		end = omp_get_wtime();
		std::cout << "Parallel with reduction: " << end - start << "s\n\n";
	}
}