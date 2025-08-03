#include <stdio.h>
#include <omp.h>

double calc(uint32_t x_last, uint32_t num_threads)
{
// ряд выглядит как : 1/1! + 1/2! + 1/3! + 1/4! = 1/0! (1/1 + 1/1*2 + 1/1*2*3 + 1/1*2*3*4)
//                    1/5! + 1/6! + 1/7! + 1/8! = 1/4! (1/5 + 1/5*6 + 1/5*6*7 + 1/5*6*7*8)
//                    1/9! + 1/10! + 1/11! + 1/12! = 1/8! (1/9 + 1/9*10 + 1/9*10*11 + 1/9*10*11*12)
// и т.д
// можно заметить, что произведение последних слагаемых в первой и во второй строчках как раз дают множитель в третьей (1/8!)
// аналогично с 1/12! - его будут давать произведение последних 3 слагаемых и т.д
// количество слагаемых зависит от количества потоков, выше был просто пример
// создадим массив для хранения значения в скобках 
double* loc_res = (double*)calloc(num_threads, sizeof(double));
// создадим массив для хранения тех величин, которые потом дадут нам факториал числа, как в примере
double* loc_fact = (double*)malloc(num_threads * sizeof(double));

#pragma omp parallel num_threads(num_threads) 
{
int tid = omp_get_thread_num();
loc_fact[tid] = 1.0;
#pragma omp for 
for (uint32_t i = 1; i < x_last; i++) {
// последнее слагаемое в скобках записываем в массив
loc_fact[tid] = loc_fact[tid]/(double)i;
// записываем сумму в массив
loc_res[tid] += loc_fact[tid];
} 
}

double res = 1.0;
double fact = 1.0;

//считаем результат 
for (uint32_t tid = 0; tid < num_threads; tid++) {
res += loc_res[tid] * fact;
fact *= loc_fact[tid];
}

free(loc_fact);
free(loc_res);
return res;
}

int main(int argc, char** argv)
{
uint32_t x_last = atoi(argv[1]);
uint32_t num_threads = atoi(argv[2]);
double res = calc(x_last, num_threads);
printf("%lf\n", res);
return 0;
}
