#include <omp.h>
#include <stdio.h>
int main(int argc, char **argv) {
// считываем аргумент из командной строки 
int N = atoi(argv[1]);
int sum = 0;
#pragma omp parallel
{	
// локальная переменная для каждого потока, 
//в которую каждый поток складывает сумму своего диапазона чисел
int local_sum = 0;
// в параллельном форе каждый поток получает свой диапазон чисел
#pragma omp for
for (int i = 1; i <= N; i++) {
local_sum +=i;
}
// атомарное суммирование локальных сумм в конечную общую
#pragma omp atomic
sum += local_sum;
}
printf("%d\n", sum);
}