#include <omp.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>

using namespace std;

void luDecomposition(std::vector<std::vector<double>> A, std::vector<std::vector<double>>& L, std::vector<std::vector<double>>& U, std::vector<double> b) {
    int n = A.size();
    #pragma omp for
    for(int k = 0; k < n; k++) {
        for(int i = k+1; i < n; i++) {
                    #pragma omp critical
            if (A[k][k] == 0.0) {
                A[k][k] = std::numeric_limits<double>::epsilon();
            }
                        #pragma omp critical
            L[i][k] = A[i][k] / A[k][k];
                        #pragma omp critical
            U[k][i] = A[k][i];
            for(int j = k+1; j < n; j++) {
                        #pragma omp critical
                A[i][j] -= L[i][k] * U[k][j];
            }
        }
    }

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < i; j++) {
            for(int k = 0; k < j; k++) {
                        #pragma omp critical
                b[i] -= L[i][k] * b[k];
            }
        }
    }

    for(int i = n-1; i >= 0; i--) {
        for(int j = i+1; j < n; j++) {
            for(int k = i+1; k < n; k++) {
                        #pragma omp critical
                b[i] -= U[i][k] * b[k];
            }
        }
                    #pragma omp critical
        if (U[i][i] == 0.0) {
            U[i][i] = std::numeric_limits<double>::epsilon();
        }
                    #pragma omp critical
        b[i] = b[i] / U[i][i];
    }
}

void luDecompositionSerial(std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& L, std::vector<std::vector<double>>& U, std::vector<double>& b) {
        int n = A.size();
    for(int k = 0; k < n; k++) {
        for(int i = k+1; i < n; i++) {
            if (A[k][k] == 0.0) {
                A[k][k] = std::numeric_limits<double>::epsilon();
            }
            L[i][k] = A[i][k] / A[k][k];
            U[k][i] = A[k][i];
            for(int j = k+1; j < n; j++) {
                A[i][j] -= L[i][k] * U[k][j];
            }
        }
    }

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < i; j++) {
            b[i] -= L[i][j] * b[j];
        }
    }

    for(int i = n-1; i >= 0; i--) {
        for(int j = i+1; j < n; j++) {
            b[i] -= U[i][j] * b[j];
        }
        if (U[i][i] == 0.0) {
            U[i][i] = std::numeric_limits<double>::epsilon();
        }
        b[i] = b[i] / U[i][i];
    }
}

int main() {
    int n = 3;

    std::vector<std::vector<double>> A(n, std::vector<double>(n));
    std::vector<double> b(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(1.0, 10.0);

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            A[i][j] = dis(gen);
        }
        b[i] = dis(gen);
    }

    std::vector<std::vector<double>> L(n, std::vector<double>(n));  // Initialize with zeros
    std::vector<std::vector<double>> U(n, std::vector<double>(n));  // Initialize with zeros
	
    // Print the input matrix A
    std::cout << "Input Matrix (A):\n";
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            std::cout << std::fixed << std::setprecision(4) << A[i][j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // Perform LU decomposition
    auto startParallel = chrono::high_resolution_clock::now();
    luDecomposition(A, L, U, b);
    auto endParallel = chrono::high_resolution_clock::now();
    chrono::duration<double> durationParallel = endParallel - startParallel;
    double executionTimeParallel = durationParallel.count();
    cout<<"Time Taken using Parallel: "<<executionTimeParallel<<endl;
	cout<<"-------------- USING PARALLEL ------------"<<endl;
    std::cout << "Lower triangular matrix (L):\n";
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if (i > j) {
                std::cout << std::fixed << std::setprecision(4) << L[i][j] << " ";
            } else if (i == j) {
                std::cout << "1.0000 ";
            } else {
                std::cout << "0.0000 ";
            }
        }
        std::cout << "\n";
    }

    std::cout << "\nUpper triangular matrix (U):\n";
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if (i <= j) {
                std::cout << std::fixed << std::setprecision(4) << U[i][j] << " ";
            } else {
                std::cout << "0.0000 ";
            }
        }
        std::cout << "\n";
    }
	cout<<"--------------USING SERIAL-----------------"<<endl;
    vector<vector<double>> LL(n, std::vector<double>(n));  // Initialize with zeros
    vector<vector<double>> UU(n, std::vector<double>(n));  // Initialize with zeros
    

    auto startSerial = chrono::high_resolution_clock::now();
    luDecompositionSerial(A, LL, UU, b);
    auto endSerial = chrono::high_resolution_clock::now();
    chrono::duration<double> durationSerial = endSerial - startSerial;
    double executionTimeSerial = durationSerial.count();
    cout<<"Time Taken using Serial: "<<executionTimeSerial<<endl;

    std::cout << "Lower triangular matrix (L):\n";
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if (i > j) {
                std::cout << std::fixed << std::setprecision(4) << LL[i][j] << " ";
            } else if (i == j) {
                std::cout << "1.0000 ";
            } else {
                std::cout << "0.0000 ";
            }
        }
        std::cout << "\n";
    }

    std::cout << "\nUpper triangular matrix (U):\n";
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            if (i <= j) {
                std::cout << std::fixed << std::setprecision(4) << UU[i][j] << " ";
            } else {
                std::cout << "0.0000 ";
            }
        }
        std::cout << "\n";
    }

//    std::cout << "\nSolution vector (x):\n";
//    for(int i = 0; i < n; i++) {
//        std::cout << "x[" << i << "] = " << std::fixed << std::setprecision(4) << b[i] << "\n";
   // }

    return 0;
}

