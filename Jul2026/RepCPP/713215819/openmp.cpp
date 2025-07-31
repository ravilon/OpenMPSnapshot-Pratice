#include <iostream>
#include <limits.h>
#include <vector>
#include <queue>
#include <omp.h>
#include <bits/stdc++.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

#define LL long long

bool parallelBFS(vector<vector<LL>>& rGraph, int s, int t, vector<LL>& parent) {
    LL V = rGraph.size();
    bool visited[V];
    fill(visited, visited + V, false);
    queue<LL> q;
    q.push(s);
    visited[s] = true;
    parent[s] = -1;
    LL stop = 0;
    while (!q.empty()) {
        LL u = q.front();
        q.pop();
        #pragma omp parallel for
        for (LL v = 0; v < V; v++) {
            if (!visited[v] && rGraph[u][v] > 0) {
                #pragma omp critical
                if (v == t) {
                    parent[v] = u;
                    stop = 1;
                }
                q.push(v);
                parent[v] = u;
                visited[v] = true;
            }
        }
        if(stop==1) return true;
    }

    return false;
}

vector<vector<LL>> generateRandomMatrix(LL size) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(1, 20);

    vector<vector<LL>> matrix(size, vector<LL>(size));

    for (LL i = 0; i < size; i++) {
        for (LL j = 0; j < size; j++) {
            matrix[i][j] = dis(gen);
        }
    }

    return matrix;
}

LL parallelFordFulkerson(vector<vector<LL>>& graph, int s, int t) {
    LL V = graph.size();

    vector<vector<LL>> rGraph(V, vector<LL>(V));
    #pragma omp parallel for
    for (LL u = 0; u < V; u++) {
        for (LL v = 0; v < V; v++) {
            rGraph[u][v] = graph[u][v];
        }
    }

    vector<LL> parent(V);

    LL max_flow = 0;

    while (parallelBFS(rGraph, s, t, parent)) {
        LL path_flow = LLONG_MAX;
        LL temp = 0;

        for (LL v = t; v != s; v=v) {
            LL u = parent[v];
            LL flow = rGraph[u][v];
            if (flow < path_flow) path_flow = flow;
            v = u;
        }

        for (LL v = t; v != s;) {
            LL u = parent[v];
            rGraph[u][v] -= path_flow;

            rGraph[v][u] += path_flow;
            v = u;
        }

        max_flow += path_flow;
    }

    return max_flow;
}
int main() {
	cout<<"---Dhivyesh R K---2021BCS0084---"<<endl;
    	cout<<"---Anirudh Gupta---2021BCS0120---"<<endl;
    	
	int choice = 0;
	cout<<"Enter 1 to give custom input"<<endl;
	cout<<"Enter 2 for testing with multiple sizes"<<endl;
	cout<<"Enter : ";
	cin>>choice;
	if(choice == 1){
		int n;
		cout<<"Enter size: ";
		cin>>n;
		vector<vector<LL>> graph(n, vector<LL>(n));
		cout<<"Enter the matrix: "<<endl;
		for(int i=0;i<n;i++){
			for(int j=0;j<n;j++){
				cin>>graph[i][j];
			}
		}
		auto start = high_resolution_clock::now();

		int output = parallelFordFulkerson(graph, 0, n - 1);

        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
		cout<<"Maximum Flow : "<<output<<endl;
        cout << "Size :  " << n << " | Execution Time:   " << duration.count() << " microseconds" << endl;
		
	}
	else{
		vector<LL> sizes = {5, 7, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 62, 63, 64};
		for (LL size : sizes) {
			
		    vector<vector<LL>> graph = generateRandomMatrix(size);

		    auto start = high_resolution_clock::now();

		    parallelFordFulkerson(graph, 0, size - 1);

		    auto stop = high_resolution_clock::now();
		    auto duration = duration_cast<microseconds>(stop - start);

		    cout << "Size :  " << size << " | Execution Time:  " << duration.count() << " microseconds" << endl;
    	}
    }

    return 0;
}

