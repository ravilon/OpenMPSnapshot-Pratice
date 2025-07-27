/*
 * 0. As entradas diagonais na matriz de coeficientes são n/10. As outras
 * entradas são números decimais aleatórios entre 0 e 1.
 *
 * 1. A solução para o sistema é x[i] = 1.
 *
 * 2. A eliminação gaussiana sobrescreve A com uma matriz triangular superior.
 *
 * 3. Presume-se que as trocas de linhas não são necessárias.
 *
 * 4. A diretiva schedule(runtime) é usada para os loops paralelizados.
 * Portanto, a variável de ambiente OMP_SCHEDULE deve ser configurada.
 */

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void get_argumentos(int argc, char* argv[], int* thread_count_p, int* n_p);
void inicializar(double A[], double b[], double x[], int n);
void eliminacao_gaussiana(double A[], double b[], int n, int thread_count);
void computar_linha(double A[], double b[], double x[], int n,
                    int thread_count);
void computar_coluna(double A[], double b[], double x[], int n,
                     int thread_count);
double achar_erro(double x[], int n);
void printar_matriz(char title[], double A[], int n);
void pritar_vetor(char title[], double x[], int n);

int main(int argc, char* argv[]) {
    int n, thread_count;
    double *A, *b, *x;
    double ge_start, ge_finish, bs_start, bs_finish, start, finish;

    get_argumentos(argc, argv, &thread_count, &n);

    A = malloc(n * n * sizeof(double));
    b = malloc(n * sizeof(double));
    x = malloc(n * sizeof(double));

    inicializar(A, b, x, n);

    ge_start = start = omp_get_wtime();
    eliminacao_gaussiana(A, b, n, thread_count);
    ge_finish = bs_start = omp_get_wtime();

    computar_linha(A, b, x, n, thread_count);
    bs_finish = finish = omp_get_wtime();
    printf("Max error in solution = %e\n", Find_error(x, n));
    printf("Time for Gaussian elim = %e seconds\n", ge_finish - ge_start);
    printf("Time for back sub = %e seconds\n", bs_finish - bs_start);
    printf("Total time for solve = %e seconds\n", finish - start);

    free(A);
    free(b);
    free(x);

    return 0;
}

void get_argumentos(int argc, char* argv[], int* thread_count_p, int* n_p) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <thread_count> <n>\n", argv[0]);
        exit(0);
    }
    *thread_count_p = strtol(argv[1], NULL, 10);
    *n_p = strtol(argv[2], NULL, 10);
}

void inicializar(double A[], double b[], double x[], int n) {
    int i, j;

    for (i = 0; i < n; i++) x[i] = 1.0;

    srandom(1);
    memset(A, 0, n * n * sizeof(double));
    for (i = 0; i < n; i++) {
        A[i * n + i] = n / 10.0;
        for (j = i + 1; j < n; j++)
            A[i * n + j] = random() / ((double)RAND_MAX);
    }

    for (i = 0; i < n; i++) {
        b[i] = 0;
        for (j = i; j < n; j++) b[i] += A[i * n + j] * x[j];
    }

    memset(x, 0, n * sizeof(double));
}

/*
 * 1. O loop externo não pode ser paralelizado devido a uma dependência entre
 * iterações: x[i] depende de x[j] para j = i+1, i+2, ..., n-1.
 * 2. O loop interno pode ser paralelizado: é apenas uma redução. Observe o uso
 * das diretivas únicas. Elas garantem que a inicialização de tmp e a atribuição
 * a x[i] sejam executadas apenas por uma thread. Além disso, como possuem
 * barreiras implícitas, garantem que nenhuma thread pode começar a executar o
 * loop interno até que a inicialização seja concluída, e nenhuma thread pode
 * iniciar uma iteração subsequente do loop externo até que x[i] tenha sido
 * calculado.
 * 3. Observe que um array não pode ser uma variável de redução. Portanto, o uso
 * de tmp é necessário.
 */
void computar_linha(double A[], double b[], double x[], int n,
                    int thread_count) {
    int i, j;
    double tmp;

#pragma omp parallel num_threads(thread_count) default(none) private(i, j) \
    shared(A, b, n, x, tmp)
    for (i = n - 1; i >= 0; i--) {
#pragma omp single
        tmp = b[i];
#pragma omp for reduction(+ : tmp) schedule(runtime)
        for (j = i + 1; j < n; j++) tmp += -A[i * n + j] * x[j];
#pragma omp single
        { x[i] = tmp / A[i * n + i]; }
    }
}

/*
 * 1. O (segundo) loop externo possui uma dependência de dados entre iterações.
 * O valor de x[j] na iteração atual geralmente foi alterado em iterações
 * anteriores. Também haverá uma condição de corrida nas atualizações de x[i]:
 * se as iterações forem divididas entre as threads, várias threads podem tentar
 * atualizar x[i] simultaneamente com seus valores de x[j].
 *
 * 2.  No entanto, as iterações no loop interno são independentes, desde que
 * todas as threads estejam trabalhando com o mesmo x[j]. Mais uma vez, observe
 * o uso da diretiva única.
 */
void computar_coluna(double A[], double b[], double x[], int n,
                     int thread_count) {
    int i, j;

#pragma omp parallel num_threads(thread_count) default(none) private(i, j) \
    shared(A, b, x, n)
    {
#pragma omp for
        for (i = 0; i < n; i++) x[i] = b[i];

        for (j = n - 1; j >= 0; j--) {
#pragma omp single
            x[j] /= A[j * n + j];
#pragma omp for schedule(runtime)
            for (i = 0; i < j; i++) x[i] += -A[i * n + j] * x[j];
        }
    }
}

double achar_erro(double x[], int n) {
    int i;
    double error = 0.0, tmp;

    for (i = 0; i < n; i++) {
        tmp = fabs(x[i] - 1.0);
        if (tmp > error) error = tmp;
    }
    return error;
}

void printar_matriz(char title[], double A[], int n) {
    int i, j;

    printf("%s:\n", title);
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) printf("%4.1f ", A[i * n + j]);
        printf("\n");
    }
    printf("\n");
}

void pritar_vetor(char title[], double x[], int n) {
    int i;

    printf("%s ", title);
    for (i = 0; i < n; i++) printf("%.1f ", x[i]);
    printf("\n");
}