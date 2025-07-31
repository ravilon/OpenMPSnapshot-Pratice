// Calculate the value of pi using OpenMP in C++

#include <chrono>
#include <iostream>
#include <string>

using namespace std;


int main(int argc, char **argv)
{
	long long num_steps = 1000000;
	if (argc > 1) {
		num_steps = stol(argv[1]);
	}

	cout << "Calculating pi in " << num_steps << " steps..." << endl;

	auto start = chrono::steady_clock::now();

	double total = 0.0;
	#pragma omp parallel for reduction(+:total)
	for (long long i = 0; i < num_steps; ++i) {
		double x = (i + 0.5) / num_steps;
		total += 4.0 / (1.0 + x * x);
	}
	total /= num_steps;

	auto finish = chrono::steady_clock::now();
	chrono::duration<double> duration = finish - start;

	cout.precision(17);
	cout << "==> pi = " << total << "\n";
	cout.precision(6);
	cout << "Calculation took " << duration.count() << " seconds." << endl;

	return 0;
}
