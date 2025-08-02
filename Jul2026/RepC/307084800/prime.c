#include <omp.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char **argv) {
	// считываем n
    int n = atoi(argv[1]);
    // массив, если primes[i] = 0 -> i простое 
    int primes[n+1];
    // заполняем изначально нулями 
    #pragma omp parallel
    #pragma omp for
    for (int i = 0; i <= n; i++) {
        primes[i] = 0;
    }
    // реализуем алгоритм решето эратосфена
    int i = 2;
    // изначально необходимо знать для pragma omp for
    // сколько всего итераций содержит цикл, вариант
    // for (int i = 2; i * i <= n; i++) не подходит
    int lim = sqrt(n);
    #pragma omp for
    // "зачеркиваем" все числа, кратные 2, 3 и тд
    for (i = 2; i <= lim; i++) { 
        for (int j = i * i; j <= n; j += i) {
            // метка того, что число не простое
            primes[j] = 1; 
        }
    }
    // печатаем все простые числа
    #pragma omp for
    for (int i = 2; i <= n; i++) {
         if (primes[i] == 0) {
            printf("%d\n", i);
        }
    } 
}