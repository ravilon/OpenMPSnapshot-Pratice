#include "strassen.hpp"
#include <omp.h>

Matrix add(const Matrix& A, const Matrix& B){
int n = A.size();
Matrix C(n, std::vector<long double>(n, 0.0));

for(int i = 0; i < n; ++i) {
for(int j = 0; j < n; ++j) {
C[i][j] = A[i][j] + B[i][j];
}
}
return C;
}

Matrix subtract(const Matrix& A, const Matrix& B){
int n = A.size();
Matrix C(n, std::vector<long double>(n, 0.0));

for(int i = 0; i < n; ++i) {
for(int j = 0; j < n; ++j) {
C[i][j] = A[i][j] - B[i][j];
}
}
return C;
}

Matrix strassen(const Matrix& A, const Matrix& B) {
int n = A.size();
Matrix C(n, std::vector<long double>(n, 0.0));

if(n == 1){
C[0][0] = A[0][0] * B[0][0];
return C;
}

int mid = n / 2;

Matrix A11(mid, std::vector<long double>(mid)), 
A12(mid, std::vector<long double>(mid)),
A21(mid, std::vector<long double>(mid)),
A22(mid, std::vector<long double>(mid)),
B11(mid, std::vector<long double>(mid)),
B12(mid, std::vector<long double>(mid)),
B21(mid, std::vector<long double>(mid)),
B22(mid, std::vector<long double>(mid));

for(int i = 0; i < mid; ++i) {
for(int j = 0; j < mid; ++j) {
A11[i][j] = A[i][j];
A12[i][j] = A[i][j + mid];
A21[i][j] = A[i + mid][j];
A22[i][j] = A[i + mid][j + mid];

B11[i][j] = B[i][j];
B12[i][j] = B[i][j + mid];
B21[i][j] = B[i + mid][j];
B22[i][j] = B[i + mid][j + mid];
}
}

Matrix P, Q, R, S, T, U, V;

#pragma omp parallel sections 
{
#pragma omp section
{
P = strassen(add(A11, A22), add(B11, B22));
}

#pragma omp section
{
Q = strassen(add(A21, A22), B11);
}

#pragma omp section
{
R = strassen(A11, subtract(B12, B22));
}
#pragma omp section
{
S = strassen(A22, subtract(B21, B11));
}
#pragma omp section
{
T = strassen(add(A11, A12), B22);
}
#pragma omp section
{
U = strassen(subtract(A21, A11), add(B11, B12));
}
#pragma omp section
{
V = strassen(subtract(A12, A22), add(B21, B22));
}
}





Matrix C1 = add(subtract(add(P, S), T), V);
Matrix C2 = add(R, T);
Matrix C3 = add(Q, S);
Matrix C4 = add(subtract(add(P, R), Q), U);

for(int i = 0; i < mid; ++i) {
for(int j = 0; j < mid; ++j) {
C[i][j] = C1[i][j];
C[i][j + mid] = C2[i][j];
C[i + mid][j] = C3[i][j];
C[i + mid][j + mid] = C4[i][j];
}
}

return C;

}