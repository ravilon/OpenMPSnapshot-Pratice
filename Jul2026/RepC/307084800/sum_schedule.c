#include <omp.h>
#include <stdio.h>
int main(int argc, char **argv) {
// считываем из консоли
int N = atoi(argv[1]);
int sum = 0;
// статическое распределение интераций - одинаковое на каждый поток
#pragma omp parallel for schedule(static)
for (intptr_t i = 0; i <= N; i++) {
sum = sum + i;
}

printf("%d\n", sum);
}