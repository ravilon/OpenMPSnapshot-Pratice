#include <stdio.h>
#include <omp.h>

int main() {
int condition=0;

#pragma omp parallel if (condition)

{
printf("Hello\n");
}
}
