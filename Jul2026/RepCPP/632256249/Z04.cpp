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

	std::vector<int> dims{ 10, 50, 100, 500, 1500, 5000, 20000, 500000 };
	for (const int& n : dims) {
		std::vector<int> vec(n);
		init_vec(vec);

		double start, end, avg_serial, avg_parallel;

		std::cout << "N: " << n << "\n";

		long sum = 0;
		start = omp_get_wtime();
		{
			for (int i = 0; i < n; i++) {
				sum += vec[i];
			}
			avg_serial = static_cast<double>(sum) / n;
		}
		end = omp_get_wtime();
		std::cout << "Serial: " << end - start << "s\n";

		sum = 0;
		start = omp_get_wtime();
		{
			#pragma omp parallel for
			for (int i = 0; i < n; i++) {
				#pragma omp critical
				sum += vec[i];
			}
			avg_parallel = static_cast<double>(sum) / n;
		}
		end = omp_get_wtime();
		std::cout << "Parallel: " << end - start << "s\n";

		sum = 0;
		start = omp_get_wtime();
		{
			#pragma omp parallel for reduction(+:sum)
			for (int i = 0; i < n; i++) {
				sum += vec[i];
			}
			avg_parallel = static_cast<double>(sum) / n;
		}
		end = omp_get_wtime();
		std::cout << "Parallel with reduction: " << end - start << "s\n\n";
	}
}