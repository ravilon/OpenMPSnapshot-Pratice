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

	std::vector<int> dims{ 10, 50, 150, 500 };
	for (const int& n : dims) {
		size_t size = n * n;
		std::vector<int> a(size), b(size), c_serial(size), c_parallel(size);
		init_vec(a);
		init_vec(b);

		double start, end;

		std::cout << "N: " << n << "\n";

		start = omp_get_wtime();
		{
			for (long i = 0; i < n; i++) {
				for (long j = 0; j < n; j++) {
					c_serial[i * n + j] = 0;
					for (long k = 0; k < n; k++) {
						c_serial[i * n + j] += a[i * n + k] * b[k * n + j];
					}
				}
			}
		}
		end = omp_get_wtime();
		std::cout << "Serial: " << end - start << "s\n";

		start = omp_get_wtime();
		{
			#pragma omp parallel for 
			for (long i = 0; i < size; i++) {
				int prod = 0;
				for (long k = 0; k < n; k++) {
					prod += a[i / n * n + k] * b[k * n + i % n];
				}
				c_parallel[i] = prod;
			}

			//for (long i = 0; i < n; i++) {
			//	for (long j = 0; j < n; j++) {
			//		int prod = 0;
			//		for (long k = 0; k < n; k++) {
			//			prod += a[i * n + k] * b[k * n + j];
			//		}
			//		c_parallel[i * n + j] = prod;
			//	}
			//}
		}
		end = omp_get_wtime();
		std::cout << "Parallel: " << end - start << "s\n";

		std::cout << (c_serial == c_parallel ? "CORRECT" : "INCORRECT") << "\n\n";
	}
}