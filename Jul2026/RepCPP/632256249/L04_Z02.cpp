#include <iostream>
#include <omp.h>
#include <vector>
#include <utility>

std::vector<int> create_vec(size_t size) {
	std::vector<int> vec;
	vec.reserve(size);

	for (int i = 0; i < vec.capacity(); i++) {
		vec.push_back(i + 1);
	}

	return vec;
}

int main()
{
	int num_threads = omp_get_num_procs();
	omp_set_num_threads(num_threads);
	std::cout << "Number of threads: " << num_threads << "\n\n";

	std::cout << std::fixed;

	std::vector<std::pair<int, int>> dims = { { 1000, 1200}, { 2500, 2000 }, { 3000, 4000 } };
	for (const auto& dim : dims) {
		int m = dim.first, n = dim.second;

		size_t size = n * m;

		std::cout << "M: " << m << ", N: " << n << "\n";

		auto mat_serial = create_vec(size);
		auto mat_parallel = mat_serial;
		auto mat_parallel_swapped = mat_serial;

		double start, end;

		start = omp_get_wtime();
		{
			for (int i = 0; i < m; i++) {
				for (int j = 2; j < n; j++) {
					mat_serial[i * n + j] = 2 * mat_serial[i * n + (j - 2)];
				}
			}
		}
		end = omp_get_wtime();
		std::cout << "Serial: " << end - start << "s\n";

		start = omp_get_wtime();
		{
			#pragma omp parallel for
			for (int i = 0; i < m; i++) {
				for (int j = 2; j < n; j++) {
					mat_parallel[i * n + j] = 2 * mat_parallel[i * n + (j - 2)];
				}
			}

		}
		end = omp_get_wtime();
		std::cout << "Parallel: " << end - start << "s\n";

		start = omp_get_wtime();
		{
			for (int j = 2; j < n; j++) {
				#pragma omp parallel for
				for (int i = 0; i < m; i++) {
					mat_parallel_swapped[i * n + j] = 2 * mat_parallel_swapped[i * n + (j - 2)];
				}
			}
		}
		end = omp_get_wtime();
		std::cout << "Parallel swapped loops: " << end - start << "s\n\n";
	}
}