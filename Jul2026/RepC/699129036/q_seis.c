#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

void computar_tarefas(int thread_count, int n) {
int i;
int* min = malloc(thread_count * sizeof(int));
int* max = malloc(thread_count * sizeof(int));

for (i = 0; i < thread_count; i++) {
min[i] = n;
max[i] = 0;
}

#pragma omp parallel num_threads(thread_count) default(none) private(i)  shared(min, max, n)
{
int my_rank = omp_get_thread_num();
#pragma omp for
for (i = 0; i < n; i++) {
if (i < min[my_rank]) min[my_rank] = i;
if (i > max[my_rank]) max[my_rank] = i;
}
}

for (i = 0; i < thread_count; i++) {
if (min[i] == n && max[i] == 0)
printf("Th %d > No iterations\n", i);
else if (min[i] != max[i])
printf("Th %d > Iterations %d - %d\n", i, min[i], max[i]);
else
printf("Th %d > Iteration  %d\n", i, min[i]);
}

free(min);
free(max);
}

int main(int argc, char* argv[]) {
int thread_count, n;

thread_count = strtol(argv[1], NULL, 10);
n = strtol(argv[2], NULL, 10);

if (n % thread_count != 0) {
printf(
"A quantidade de loops deve ser igualmente divisível pela "
"quantidade de threads.\n");
exit(0);
}

computar_tarefas(thread_count, n);

exit(0);
}
