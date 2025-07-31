#include <iostream>
#include <omp.h>

int main()
{
	std::cout << std::fixed;

	int num_threads = omp_get_num_procs();
	omp_set_num_threads(num_threads);

	constexpr int n = 200;
	int x_par[n], y_par[n], g_par,
		x_seq[n], y_seq[n], g_seq,
		z[n];

	for (int i = 0; i < n; i++) {
		x_par[i] = x_seq[i] = y_par[i] = y_seq[i] = z[i] = i + 1;
	}

	double start, end;
	start = omp_get_wtime();

	g_seq = 0;
	for (int i = 1; i < n; i++) {
		y_seq[i] += x_seq[i - 1];
		x_seq[i] += z[i];
		g_seq += z[i - 1];
	}
	
	end = omp_get_wtime();
	std::cout << "Serial: " << end - start << "s\n";

	start = omp_get_wtime();

	g_par = z[0];
	y_par[1] += x_par[0];
	#pragma omp parallel for reduction(+:g_par)
	for (int i = 1; i < n - 1; i++) {
		x_par[i] += z[i];
		g_par += z[i];
		y_par[i + 1] += x_par[i];
	}
	x_par[n - 1] += z[n - 1];

	end = omp_get_wtime();
	std::cout << "Parallel: " << end - start << "s\n";

	bool match = g_seq == g_par;
	if (match) {
		for (int i = 0; i < n; i++) {
			if (
				x_seq[i] != x_par[i] ||
				y_seq[i] != y_par[i]
				)
			{
				match = false;
				break;
			}
		}
	}
	std::cout << (match ? "CORRECT" : "INCORRECT") << "\n";
}
