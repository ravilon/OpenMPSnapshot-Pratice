#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>

using namespace std;

void parallelMatrixMultiplication(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C) {
    int numRowsA = A.size();
    int numColsA = A[0].size();
    int numColsB = B[0].size();

    #pragma omp parallel for
    for (int i = 0; i < numRowsA; ++i) {
        for (int j = 0; j < numColsB; ++j) {
            int sum = 0;
            for (int k = 0; k < numColsA; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

void parallelMatrixAddition(const vector<vector<int>>& A, const vector<vector<int>>& B, vector<vector<int>>& C) {
    int numRows = A.size();
    int numCols = A[0].size();

    #pragma omp parallel for
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            C[i][j] = A[i][j] + B[i][j];
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
	vector<vector<int>> B(numRows, vector<int>(numCols,0));
	for(int i=0;i<numRows;i++)
		for(int j=0;j<numCols;j++){
			A[i][j] = i*j;
			B[i][j] = i+j;
		}

    vector<vector<int>> C(numRows, vector<int>(numCols, 0));
	
	auto start = chrono::high_resolution_clock::now();

    parallelMatrixAddition(A, B, C);

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
		printf("MATRIX B : \n");
			for (int i = 0; i < numRows; ++i) {
		    for (int j = 0; j < numCols; ++j) {
				printf("%d ", B[i][j]);
		    }	
			cout << "\n";
		}
	}
	printf("Time taken for parallel Addition is %lf microseconds\n",double(duration.count()));
    // Print the result matrix
    if(output > 0){
        printf("Addition result : \n");
		for (int i = 0; i < numRows; ++i) {
		    for (int j = 0; j < numCols; ++j) {
				printf("%d ", C[i][j]);
		    }	
			cout << "\n";
		}
	}
    start = chrono::high_resolution_clock::now();

    parallelMatrixMultiplication(A, B, C);

	end = chrono::high_resolution_clock::now();
	duration = chrono::duration_cast<chrono::microseconds>(end - start);
	printf("Time taken for parallel Multiplication is %lf microseconds\n",double(duration.count()));
	
    // Print the result matrix
    if(output > 0){
        printf("Multiplication result : \n");
		for (int i = 0; i < numRows; ++i) {
		    for (int j = 0; j < numCols; ++j) {
				printf("%d ", C[i][j]);
		    }	
			cout << "\n";
		}
	}
    return 0;
}

