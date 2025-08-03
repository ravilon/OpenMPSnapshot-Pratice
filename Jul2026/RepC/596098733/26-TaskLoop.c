/*
* 26-TaskLoop.c
*
*  Created on: 26 feb. 2023
*      Author: Jose ngel Gumiel
*/


#include <stdio.h>
#include <omp.h>

int main() {
const int N = 1000;
int a[N], b[N], c[N];

// Initialize arrays a and b
for (int i = 0; i < N; i++) {
a[i] = i;
b[i] = 2 * i;
}

#pragma omp parallel
{
#pragma omp single
{
#pragma omp taskloop
for (int i = 0; i < N; i++) {
c[i] = a[i] + b[i];
}
}
}

// Print array c
printf("Result:\n");
for (int i = 0; i < N; i++) {
printf("%d ", c[i]);
}
printf("\n");

return 0;
}
