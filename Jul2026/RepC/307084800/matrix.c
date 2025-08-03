#include <omp.h>
#include <stdio.h>
#include <math.h>

void setMatrix(int **matrix, int n, int m) {
for(int i = 0; i < n; i++) {
for(int j = 0; j < m; j++) {
matrix[i][j] = rand() % 100;
}
}
}

void printMatrix(int **matrix, int n, int m) {
int k = n > 10 ? 10 : n;
int l = m > 10 ? 10 : m;
for(int i = 0; i < k; i++) {
for(int j = 0; j < l; j++) {
printf("%d ", matrix[i][j]);
}
printf("\n");
}
printf("%s\n", "-----------");
}

int main(int argc, char **argv) {
// считываем размеры матриц из командной строки 
int n1 = atoi(argv[1]);
int n2 = atoi(argv[2]);
int m1 = atoi(argv[3]);
int m2 = atoi(argv[4]);
// если матрицы не согласованы
if (n1 != m2) {
printf("%s\n", "Матрицы не согласованы! Количество строк первой должно равняться количеству столбцов второй.");
return -1;
}
//Матрица n1 x m1
int **matrix1;
//Матрица n2 x m2
int **matrix2;

matrix1 = (int**)malloc(sizeof(int*)*n1);
for(int i = 0; i < n1; i++) {
matrix1[i] = (int*)malloc(sizeof(int)*n2);
}
matrix2 = (int**)malloc(sizeof(int*)*m1);
for(int i = 0; i < m1; i++) {
matrix2[i] = (int*)malloc(sizeof(int)*m2);
}
// матрица-результат
int **res = (int**)malloc(sizeof(int*)*n1);;
for(int i = 0; i < n1; i++) {
res[i] = (int*)malloc(sizeof(int)*m2);
}
// заполняем рандомными значениями
setMatrix(matrix1, n1, n2);
setMatrix(matrix2, m1, m2);
// распечатываем для наглядности
printMatrix(matrix1, n1, n2);
printMatrix(matrix2, m1, m2);
int i, j, k;
#pragma omp parallel for private(i, j, k) 
{
for (i = 0; i < n1; i++) {
for (j = 0; j < m2; j++) {
res[i][j] = 0;
for (k = 0; k < m1; k++) {
res[i][j] += (matrix1[i][k] * matrix2[k][j]);
}
}
}
}
// печатаем результат
printMatrix(res, n1, m2);
free(matrix1);
free(matrix2);
}