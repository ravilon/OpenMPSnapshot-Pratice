#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>

using namespace std;


void parallelMatrixTranspose(const vector<vector<int>>& A, vector<vector<int>>& result) {
    int numRows = A.size();
    int numCols = A[0].size();

    #pragma omp parallel for
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            result[j][i] = A[i][j];
        }
    }
}

int main() {
    int numRows = 0;
    printf("Done By Dhivyesh R K 2021BCS0084 \n");
    printf("Enter number of rows for your square matrix:  ");
    scanf("%d", &numRows);
    int numCols = numRows;
    int output;
	printf("Do you want to see the resulting matrices:  ");
	scanf("%d",&output);
	vector<vector<int>> A(numRows, vector<int>(numCols,0));
	for(int i=0;i<numRows;i++)
		for(int j=0;j<numCols;j++){
			A[i][j] = i*j+j;
		}

    vector<vector<int>> C(numRows, vector<int>(numCols, 0));
	
	auto start = chrono::high_resolution_clock::now();

    parallelMatrixTranspose(A,C);

	auto end = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
	if(output>0){

		printf("MATRIX A : \n");
		for (int i = 0; i < numRows; ++i) {
		    for (int j = 0; j < numCols; ++j) {
				printf("%d ", A[i][j]);
		    }	
			cout << "\n";
		}
	}
	printf("Time taken for Transpose is %lf microseconds\n",double(duration.count()));
    // Print the result matrix
    if(output > 0){
        printf("Transpose result : \n");
		for (int i = 0; i < numRows; ++i) {
		    for (int j = 0; j < numCols; ++j) {
				printf("%d ", C[i][j]);
		    }	
			cout << "\n";
		}
	}
    return 0;
}

