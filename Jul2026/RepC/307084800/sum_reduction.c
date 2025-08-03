#include <omp.h>
#include <stdio.h>
int main(int argc, char **argv) {
// считываем аргумент из командной строки 
int N = atoi(argv[1]);
int sum = 0;
#pragma omp parallel for reduction(+:sum) // операция суммирования
for (intptr_t i = 0; i <= N; i++) {
sum = sum + i;
}
printf("%d\n", sum);
}