#include <omp.h>
#include <stdio.h>

/* Print string and id of current thread. */
static void myprint(const int id, const char* string) {
  printf("[%2d] %s", id, string);
  fflush(stdout);
}

/*
 * Fork and print to screen.
 */
int main () {
  #pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    myprint(thread_id, "One\n");
    myprint(thread_id, "Two\n");
    myprint(thread_id, "Three\n");
    myprint(thread_id, "Four!\n");
  }
}
