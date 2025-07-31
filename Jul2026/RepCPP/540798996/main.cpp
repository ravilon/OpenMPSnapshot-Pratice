#include <iostream>

#include <omp.h>
int main () {
/* Выделение параллельного фрагмента*/
#pragma omp parallel
{
printf("Hello World !\n");
}/* Завершение параллельного фрагмента */
}
