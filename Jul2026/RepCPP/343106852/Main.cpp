#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <chrono>
#include <omp.h>
#include <set>

using namespace std;

typedef std::chrono::duration<double> fsec;

int main() {
	omp_set_num_threads(omp_get_max_threads());
	uint64_t t, t1, t2, k;
	stringstream ss, title;;

	auto start = chrono::high_resolution_clock::now();

#pragma omp parallel for private(t, t1, t2, k) 
	for (int64_t i = 9; i <= UINT32_MAX; i++) {
		t = (uint64_t)i * (uint64_t)i;
		k = 10;

		do {
			t1 = t / k;
			t2 = t % k;
			k *= 10;
		} while (t1 > (uint64_t)i);

		while (t1 + t2 != i && t1 % 10 == 0)
			t1 /= 10;

		if (t1 + t2 == i)
			ss << setw(10) << i << ' ' << setw(20) << t << ' ' << setw(10) << t1 << ' ' << t2 << endl;
	}

	set<string> lines;
	string line;
	while (getline(ss, line))
		lines.insert(line);

	ofstream ofs("kaprekar.txt");
	for (const auto& l : lines) {
		cout << l << endl;
		ofs << l << endl;
	}
	ofs.close();

	auto end = std::chrono::high_resolution_clock::now();
	fsec duration = end - start;
	cout << "Ended for " << duration.count() << " seconds\n";
	system("pause");

	return 0;
}