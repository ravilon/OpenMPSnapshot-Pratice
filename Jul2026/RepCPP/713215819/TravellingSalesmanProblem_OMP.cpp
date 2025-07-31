#include <bits/stdc++.h>
#include <omp.h>
#include <chrono>

using namespace std;
#define V 4

// implementation of traveling Salesman Problem
int travllingSalesmanProblemParallel(int graph[][V], int s)
{
	// store all vertex apart from source vertex
	vector<int> vertex;
	for (int i = 0; i < V; i++)
		if (i != s)
			vertex.push_back(i);

	// store minimum weight Hamiltonian Cycle.
	int min_path = INT_MAX;
	#pragma omp parallel
	do {

		// store current Path weight(cost)
		int current_pathweight = 0;

		// compute current path weight
		int k = s;
		#pragma omp parallel for
		for (int i = 0; i < vertex.size(); i++) {
			current_pathweight += graph[k][vertex[i]];
			k = vertex[i];
		}
		current_pathweight += graph[k][s];

		// update minimum
		min_path = min(min_path, current_pathweight);

	} while (
		next_permutation(vertex.begin(), vertex.end()));

	return min_path;
}

int travllingSalesmanProblemSerial(int graph[][V], int s)
{
	// store all vertex apart from source vertex
	vector<int> vertex;
	for (int i = 0; i < V; i++)
		if (i != s)
			vertex.push_back(i);

	// store minimum weight Hamiltonian Cycle.
	int min_path = INT_MAX;
	do {

		// store current Path weight(cost)
		int current_pathweight = 0;

		// compute current path weight
		int k = s;
		for (int i = 0; i < vertex.size(); i++) {
			current_pathweight += graph[k][vertex[i]];
			k = vertex[i];
		}
		current_pathweight += graph[k][s];

		// update minimum
		min_path = min(min_path, current_pathweight);

	} while (
		next_permutation(vertex.begin(), vertex.end()));

	return min_path;
}

// Driver Code
int main()
{
	// matrix representation of graph
	int graph[][V] = { { 0, 10, 15, 20 },
					{ 10, 0, 35, 25 },
					{ 15, 35, 0, 30 },
					{ 20, 25, 30, 0 } };
	int graph2[V][V];
	for(int i=0;i<V;i++){
		for(int j=0;j<V;j++){
			graph2[i][j]=graph[i][j];
		}
	}
	int s = 0;
	
	auto  startParallel = chrono::high_resolution_clock::now();
	cout<<"Result using Parallel" << travllingSalesmanProblemParallel(graph2, s) << endl;
	auto endParallel = chrono::high_resolution_clock::now();
	chrono::duration<double> durationParallel = endParallel - startParallel;
	double executionTimeParallel = durationParallel.count();
	cout<<"Time Taken using Parallel: "<<executionTimeParallel<<endl;
		
	
	auto  startSerial = chrono::high_resolution_clock::now();
	cout<<"Result using Serial" << travllingSalesmanProblemSerial(graph, s) << endl;
	auto endSerial = chrono::high_resolution_clock::now();
	chrono::duration<double> durationSerial = endSerial - startSerial;
	double executionTimeSerial = durationSerial.count();
	cout<<"Time Taken using Serial : "<<executionTimeSerial<<endl;
	
	
	
	return 0;
}

