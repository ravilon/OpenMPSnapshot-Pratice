#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define U(i,j)        (u[i][j])
#define Uold(i,j)  (uold[i][j])
#define F(i,j)        (f[i][j])

void jacobi (int n, double dx, double omega, double **u, double **f, double tol, int maxit);

double** dalloc_matrix(int n, int m) {
double **mat = (double**) malloc (n * sizeof(double*));
#pragma omp parallel for
for(int i = 0; i < n; i++)
mat[i] = (double*) malloc (m * sizeof(double));
return mat;
}


void dfree_matrix(double **mat, int n) {
#pragma omp parallel for
for(int i = 0; i < n; ++i)
free(mat[i]);
free(mat);
}

double cpu_time() {
return clock()/((double) CLOCKS_PER_SEC);
}

/*****************************************************
* Inicializando os dados
******************************************************/
void initialize(int n,
double *dx,
double *dy,
double **u,
double **f) {
int i, j;
double xxexp, yy;

*dx = 1.0 / (n - 1);
*dy = 1.0 / (n - 1);

/* Inicializando condicao inicial e RHS */
#pragma omp parallel for default(shared) private(i, j, xxexp, yy)
for (i = 0; i < n; i++) {
xxexp = exp((*dx) * (i - 1));
for (j = 0; j < n; j++) {
yy = (*dy) * (j - 1);
U(i,j) = 0.0;
F(i,j) = xxexp * yy;
}
}

}

/************************************************************
* Verificacao do erro entre a solucao numerica e exata
************************************************************/
void error_check(int n,
double dx,
double dy,
double **u) {
int i, j;
double xxexp, yy, temp, error;

#pragma omp parallel for reduction(+:error) private(i, j, xxexp, yy, temp)
for (i = 0; i < n; i++){
xxexp = exp(dx * (i - 1));
for (j = 0; j < n; j++){
yy = dy * (j - 1);
temp = U(i,j) - xxexp * exp(-2.0 * yy);
error += temp * temp;
}
}

error = sqrt(error)/(n * n);
printf("Erro : %g\n", error);

}

int main(int argc, char **argv) {
int n, m, mits;
double tol, relax;
double **u, **f, dx, dy;
double dt, start, end, mflops;

m = n = 100;
relax = 1.0;
tol   = 1.0e-5;
mits  = 1000;

if(argc == 4) {
n     = atoi(argv[1]);
tol   = atof(argv[2]);
mits  = atoi(argv[3]);
} else {
printf("Use: ./jacobi <n> <tol> <mits>\n");
printf("where\n");
printf("   <n>     : numero de pontos em X e Y (default 100)\n"); 
printf("   <tol>   : tolerancia do erro (default 1e-5)\n");
printf("   <mits>  : numero maximo de iteracoes (default 1000)\n");
}

printf("-> %d, %g, %g, %d\n", n, relax, tol, mits);

// Alocando as estruturas de dados
u = dalloc_matrix(n, m);
f = dalloc_matrix(n, m);

/* Dados inicializados */
initialize(n, &dx, &dy, u, f);

double pi = acos(-1.0);

relax = 2.0/(1.0 + sin(pi*dx));

printf("relax parameter: %f \n", relax);

/* Resolvendo a equação */
start = cpu_time();
jacobi(n, dx, relax, u,f, tol, mits);
end   = cpu_time();

dt = end-start;

printf(" elapsed time : %12.6f\n", dt);
mflops = (0.000001*mits*(m-2)*(n-2)*13) / dt;
printf(" MFlops       : %12.6g (%d, %d, %d, %g)\n",mflops, mits, m, n, dt);

// Verificando o erro.
error_check(n, dx, dy, u);

// Liberando memória.
dfree_matrix(u, n);
dfree_matrix(f, n);

return 0;
}

/* 
subroutine jacobi (n,dx,alpha,omega,u,f,tol,maxit)
******************************************************************
* Resolve equação de poisson em um grid retangular assumindo : 
* (1) discretização uniforme em cada direção, e 
* (2) condiçoes de contorno de Dirichlect 
* 
* Metodo de jacobi é usado nesta rotina
*
* Input : n    número de pontos nas direcoes X/Y 
*         dx espacamento entre os pontos 
*         omega  fator de relaxação SOR 
*         f(n,m) vetor RHS 
*         u(n,m) vetor soluçao
*         tol    tolerancia do metodo iterativo
*         maxit  numero máximo de iteracoes
*
* Output : u(n,m) - solucao
*****************************************************************
*/

void jacobi (int n, double dx, double omega, double **u, double **f, double tol, int maxit ) {
int i,j,k;
double error, resid, h2;
double **uold;

uold = dalloc_matrix(n,n);

h2 = (dx * dx);   /* X-direction coef */

error = 10.0 * tol;
k = 1;
while (k <= maxit && error > tol) 
{
/* copia solução corrente para solução antiga */
#pragma omp parallel for private(i, j) collapse(2)
for (i = 0; i < n; i++)
for (j = 0; j < n; j++)
Uold(i,j) = U(i,j);

/* calcula o residuo por diferencas finitas */
#pragma omp parallel for reduction(+:error) private(i, j, resid) collapse(2)
for (i = 1; i < n - 1; i++) {
for (j = 1; j < n - 1; j++) {
/* atualiza a solução */
U(i,j) = (1.0 - omega) * U(i,j) + omega * (U(i-1,j) + U(i+1,j) + U(i,j-1) + U(i,j+1) + h2 * F(i,j)) / 4.0;

/* acumula o erro relativo */
resid = U(i,j) - Uold(i,j);
error += resid * resid;
}
}

/* verifica o erro  */
k++;
error = sqrt(error);

} /* while */

printf("Numero total de iteracoes:  %d\n", k);
printf("Error                  : %10.15f\n", error);
dfree_matrix(uold, n);
} 
