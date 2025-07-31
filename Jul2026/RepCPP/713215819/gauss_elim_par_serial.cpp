#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

using namespace std;

void gaussianEliminationSerial(vector<vector<double>>& A) {
    int n = A.size();

    for (int i = 0; i < n; i++) {
        int maxRow = i;
        for (int k = i + 1; k < n; k++) {
            if (abs(A[k][i]) > abs(A[maxRow][i]))
                maxRow = k;
        }
        swap(A[i], A[maxRow]);

        for (int k = i + 1; k < n; k++) {
            double factor = A[k][i] / A[i][i];
            for (int j = i; j <= n; j++) {
                A[k][j] -= factor * A[i][j];
            }
        }
    }
}

void gaussianEliminationParallel(vector<vector<double>>& A) {
	omp_set_num_threads(3);
    int n = A.size();
    for (int i = 0; i < n; i++) {
        int maxRow = i;
        for (int k = i + 1; k < n; k++) {
            if (abs(A[k][i]) > abs(A[maxRow][i]))
                maxRow = k;
        }
        swap(A[i], A[maxRow]);
        #pragma omp parallel for 
        for (int k = i + 1; k < n; k++) {
            cout<<"Number of threads are : "<<omp_get_num_threads()<<endl;
            double factor = A[k][i] / A[i][i];
            for (int j = i; j <= n; j++) {
                    cout<<"Thread id is "<<omp_get_thread_num()<<endl;
                    #pragma omp critical
                A[k][j] -= factor * A[i][j];
            }
        }
    }
}

vector<double> backSubstitution(vector<vector<double>>& A) {
    int n = A.size();
    vector<double> x(n);

    for (int i = n - 1; i >= 0; i--) {
        x[i] = A[i][n] / A[i][i];
        for (int k = i - 1; k >= 0; k--) {
            A[k][n] -= (A[k][i] * x[i]);
        }
    }

    return x;
}

int main() {
    cout<<"Done by DHIVYESH RK 2021bcs0084"<<endl;
    int n;
    cout << "Enter the number of equations: ";
    cin >> n;

    vector<vector<double>> A(n, vector<double>(n + 1));

    cout << "Enter the coefficients of the system:" << endl;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= n; j++) {
            cin >> A[i][j];
        }
    }

    // Serial Version
    auto startSerial = chrono::high_resolution_clock::now();
    gaussianEliminationSerial(A);
    vector<double> solutionSerial = backSubstitution(A);
    auto endSerial = chrono::high_resolution_clock::now();
    chrono::duration<double> durationSerial = endSerial - startSerial;
    double executionTimeSerial = durationSerial.count();

    // Parallel Version
    vector<vector<double>> B(n, vector<double>(n + 1));
    B = A; // Copy the original matrix to B for parallel version

    auto startParallel = chrono::high_resolution_clock::now();
    gaussianEliminationParallel(B);
    vector<double> solutionParallel = backSubstitution(B);
    auto endParallel = chrono::high_resolution_clock::now();
    chrono::duration<double> durationParallel = endParallel - startParallel;
    double executionTimeParallel = durationParallel.count();
    
    // Output results
//    cout << "Serial Version:" << endl;
   // cout << "Solution: ";
   // for (double val : solutionSerial) {
   //     cout << val << " ";
   // }
    cout << endl;
    cout << "Serial Execution Time: " << executionTimeSerial << " seconds" << endl;

//    cout << "\nParallel Version:" << endl;
   // cout << "Solution: ";
   // for (double cal : solutionParallel) {
   //     cout << cal << " ";
   // }
    cout << endl;
    cout << "Parallel Execution Time: " << executionTimeParallel << " seconds" << endl;
    cout<<"Solution vector : ";
    for(double val : solutionSerial) {
    	cout<<val<<" ";
    }
    cout<<endl;

    return 0;
}

