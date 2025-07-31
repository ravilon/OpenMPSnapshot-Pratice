#include <iostream>
#include <omp.h>
#include <vector>

std::vector<int> create_vec(size_t size) {
	std::vector<int> vec;
	vec.reserve(size);

	for (int i = 0; i < vec.capacity(); i++) {
		vec.push_back(i + 1);
	}

	return vec;
}

int main() {
	int num_threads = omp_get_num_procs();
	omp_set_num_threads(num_threads);
	std::cout << "Number of threads: " << num_threads << "\n\n";

	std::cout << std::fixed;

	constexpr int n = 512, m = 256;
	size_t size = n * m;
	
	std::vector<int> mat_serial, mat_parallel, mat_parallel_swapped;
	mat_serial = create_vec(size);
	mat_parallel = create_vec(size);
	mat_parallel_swapped = create_vec(size);

	double start, end;

	start = omp_get_wtime();
	{
		for (int i = 1; i < m; i++) {
			for (int j = 0; j < n; j++) {
				mat_serial[i * n + j] = 2 * mat_serial[(i - 1) * n + j];
			}
		}
	}
	end = omp_get_wtime();
	std::cout << "Serial: " << end - start << "s\n";

	start = omp_get_wtime();
	{
		for (int i = 1; i < m; i++) {
			#pragma omp parallel for
			for (int j = 0; j < n; j++) {
				mat_parallel[i * n + j] = 2 * mat_parallel[(i - 1) * n + j];
			}
		}
	}
	end = omp_get_wtime();
	std::cout << "Parallel: " << end - start << "s\n";

	start = omp_get_wtime();
	{
		#pragma omp parallel for
		for (int j = 0; j < n; j++) {
			for (int i = 1; i < m; i++) {
				mat_parallel_swapped[i * n + j] = 2 * mat_parallel_swapped[(i - 1) * n + j];
			}
		}
	}
	end = omp_get_wtime();
	std::cout << "Parallel swapped: " << end - start << "s\n\n";
}