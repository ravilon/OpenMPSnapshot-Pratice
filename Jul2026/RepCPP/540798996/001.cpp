// Задача 1
// Разработайте программу для нахождения минимального
// (максимального) значения среди элементов вектора.

#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <time.h>
#include <stdlib.h>
#include <conio.h>
using namespace std;

int main(){

// int n = 1410065408; // 10**10
int n = 1000000000; // 10**9
// int n = 100000000; // 10**8
// int n = 1000000; // 10**6
// int n = 100000; // 10**5
// int n = 1000; // 10**3

// int *r = NULL;
// r = (int*) malloc(N * sizeof(int));
int *r = (int*) calloc(n, sizeof(int));
// free(r);

// int r[N];
double time_spent = 0.00000000;
double time_spent_par = 0.00000000;

// Генерация массива на 1000 случайный чисел
for(int i = 0; i < n; ++i) {
  r[i] = rand();
  // r[i] = 12;
  // printf("%i\n",r[i]);

}

// Задаём переменную для поиска минимального и максимального значения в массиве
int min = r[0];
int max = r[0];
int max_sum = 0;
int min_par = r[0];
int max_par = r[0];
int max_sum_par = 0;

// // Расчет времени при последовательном выполнении
clock_t begin =  clock();
for(int i = 0; i < n; ++i)
{
  if(r[i] > max)
  {
      max = r[i];
      // printf("%i\n",r[i]);
  }
  if(r[i] < min)
  {
      min = r[i];
      // printf("%i\n",r[i]);
  }
  // max_sum += r[i];
}
clock_t end =  clock();
time_spent += (double)(end - begin) / (CLOCKS_PER_SEC);
printf("\nSequential work time is %.10f seconds, max elem = %i, min elem = %i", time_spent, max, min);


// printf("%i\n", r);

// Расчёт времени при параллельном выполнении
clock_t begin_par =  clock();
#pragma omp parallel for  num_threads(10)
  for(int i = 0; i < n; ++i)
  {
    if(r[i] > max_par)
    {
        max_par = r[i];
        // printf("\n%i",r[i]);
    }
    if(r[i] < min_par)
    {
        min_par = r[i];
        // printf("%i\n",r[i]);
    }
    // max_sum_par += r[i];

  }
  clock_t end_par =  clock();
  time_spent_par += (double)(end_par - begin_par) / (CLOCKS_PER_SEC);
  printf("\nParallel work time is %.10f seconds, max elem = %i, min elem = %i", time_spent_par, max_par, min_par);
free(r);
return 0;
}