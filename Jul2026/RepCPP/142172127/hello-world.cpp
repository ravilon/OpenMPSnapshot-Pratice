
#include "bits/stdc++.h"
#include "omp.h"
using namespace std;
int main()
{
	// set default threads to 5
	omp_set_num_threads(5);

	// create parallel region
	#pragma omp parallel
	{
		printf("Hello World\n");
	}
}
