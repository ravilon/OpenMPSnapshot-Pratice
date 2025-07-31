/* File:
 *    omp_tokenize.c
 *
 * Purpose:
 *    Try to use threads to tokenize text input.  Illustrate problems
 *    with function that isn't threadsafe.  This program has a serious
 *    bug.
 *
 * Compile:
 *    gcc -g -Wall -fopenmp -o omp_tokenize omp_tokenize.c
 * Usage:
 *    omp_tokenize <thread_count> < <input>
 *
 * Input:
 *    Lines of text
 * (Desired) Output:
 *    For each line of input:
 *       the line read by the program, and the tokens identified by
 *       strtok
 *
 * Algorithm:
 *    For each line of input, next thread reads the line and
 *    "tokenizes" it.
 *
 * IPP:   Section 5.10 (pp. 256 and ff.)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

const int MAX_LINES = 1000;
const int MAX_LINE = 80;

void como_usar(char* prog_name);
void pegar_texto(char* lines[], int* line_count_p);
void tokens(char* lines[], int line_count, int thread_count);

/*--------------------------------------------------------------------*/
int main(int argc, char* argv[]) {
   int thread_count, i;
   char* lines[1000];
   int line_count;

   if (argc != 2) como_usar(argv[0]);
   thread_count = strtol(argv[1], NULL, 10);

   printf("Enter text\n");
   pegar_texto(lines, &line_count);
   tokens(lines, line_count, thread_count);

   for (i = 0; i < line_count; i++)
      if (lines[i] != NULL) free(lines[i]);

   return 0;
}  /* main */


/*--------------------------------------------------------------------
 * Function:    Usage
 * Purpose:     Print command line for function and terminate
 * In arg:      prog_name
 */
void como_usar(char* prog_name) {

   fprintf(stderr, "usage: %s <number of threads>\n", prog_name);
   exit(0);
}  /* Usage */

/*--------------------------------------------------------------------
 * Function:  Get_text
 * Purpose:   Read text and store as an array of strings, one per line
 *            of input text
 * Out args:  lines, line_count_p
 */
void pegar_texto(char* lines[], int* line_count_p) {
   char* line = malloc(MAX_LINE*sizeof(char));
   int i = 0;
   char* fg_rv;

   fg_rv = fgets(line, MAX_LINE, stdin);
   while (fg_rv != NULL) {
      lines[i++] = line;
      line = malloc(MAX_LINE*sizeof(char));
      fg_rv = fgets(line, MAX_LINE, stdin);
   }
   *line_count_p = i;
}  /* Get_text */

/*-------------------------------------------------------------------
 * Function:    Tokenize
 * Purpose:     Tokenize lines of input
 * In args:     line_count, thread_count
 * In/out arg:  lines
 */
void tokens(
      char*  lines[]       /* in/out */,
      int    line_count    /* in     */,
      int    thread_count  /* in     */) {
   int my_rank, i, j;
   char *my_token;

#  pragma omp parallel num_threads(thread_count) \
      default(none) private(my_rank, i, j, my_token) shared(lines, line_count)
   {
      my_rank = omp_get_thread_num();
#     pragma omp for schedule(static, 1)
      for (i = 0; i < line_count; i++) {
         printf("Thread %d > line %d = %s", my_rank, i, lines[i]);
         j = 0;
         my_token = strtok(lines[i], " \t\n");
         while ( my_token != NULL ) {
            printf("Thread %d > token %d = %s\n", my_rank, j, my_token);
            my_token = strtok(NULL, " \t\n");
            j++;
         }
      if (lines[i] != NULL)
         printf("Thread %d > After tokenizing, my line = %s\n",
            my_rank, lines[i]);
      } /* for i */
   }  /* omp parallel */

}  /* Tokenize */
