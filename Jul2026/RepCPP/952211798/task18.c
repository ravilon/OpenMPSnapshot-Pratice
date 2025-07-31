/*
** PROGRAMA: SOLUÇÃO DA EQUAÇÃO DO CALOR
**
** OBJETIVO: ESTE PROGRAMA EXPLORA O USO DE UM MÉTODO EXPLÍCITO
**           DE DIFERENÇAS FINITAS PARA RESOLVER A EQUAÇÃO DO CALOR
**           UTILIZANDO UM MÉTODO DE SOLUÇÃO MANUFATURADA (MMS).
**           A SOLUÇÃO É UMA FUNÇÃO SIMPLES BASEADA EM EXPONENCIAIS E FUNÇÕES TRIGONOMÉTRICAS.
**
**           O ESQUEMA É APLICADO EM UMA MALHA 1000x1000.
**           SIMULA UM TEMPO TOTAL DE 0.5 UNIDADES.
**
**           A SOLUÇÃO MMS FOI ADAPTADA DE:
**           G.W. RECKTENWALD (2011). FINITE DIFFERENCE APPROXIMATIONS TO THE HEAT EQUATION.
**           PORTLAND STATE UNIVERSITY.
**
**
** USO: EXECUTAR COM DOIS ARGUMENTOS:
**      PRIMEIRO É O NÚMERO DE CÉLULAS.
**      SEGUNDO É O NÚMERO DE PASSOS DE TEMPO.
**
**      EXEMPLO: ./heat 100 10
**
**
** HISTÓRICO: ESCRITO POR TOM DEAKIN, OUTUBRO DE 2018
**
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <omp.h>  // INCLUSÃO DA BIBLIOTECA OPENMP PARA PARALELISMO

// CONSTANTES CHAVE USADAS NO PROGRAMA
#define PI acos(-1.0) // PI DEFINIDO USANDO ARCO COSSENO
#define LINE "--------------------\n" // LINHA PARA FORMATAÇÃO DE SAÍDA

// DECLARAÇÃO DAS FUNÇÕES USADAS NO PROGRAMA
void initial_value(const int n, const double dx, const double length, double * restrict u);
void zero(const int n, double * restrict u);
void solve(const int n, const double alpha, const double dx, const double dt, const double * restrict u, double * restrict u_tmp);
double solution(const double t, const double x, const double y, const double alpha, const double length);
double l2norm(const int n, const double * restrict u, const int nsteps, const double dt, const double alpha, const double dx, const double length);


// FUNÇÃO PRINCIPAL
int main(int argc, char *argv[]) {

  // INÍCIO DA CONTAGEM DE TEMPO TOTAL DO PROGRAMA (PARA MEDIÇÃO DE DESEMPENHO)
  double start = omp_get_wtime();

  // TAMANHO DA MALHA NXN (PADRÃO 1000)
  int n = 1000;

  // NÚMERO DE PASSOS DE TEMPO (PADRÃO 10)
  int nsteps = 10;

  // VALIDAÇÃO DE ARGUMENTOS DE LINHA DE COMANDO
  // SE ARGUMENTOS SÃO PASSADOS, USAR ESSES VALORES PARA N E NSTEPS
  if (argc == 3) {
    n = atoi(argv[1]);
    if (n < 0) {
      fprintf(stderr, "Error: n must be positive\n");
      exit(EXIT_FAILURE);
    }

    nsteps = atoi(argv[2]);
    if (nsteps < 0) {
      fprintf(stderr, "Error: nsteps must be positive\n");
      exit(EXIT_FAILURE);
    }
  }

  // DEFINIÇÕES DE PARÂMETROS DO PROBLEMA
  double alpha = 0.1;          // COEFICIENTE DA EQUAÇÃO DO CALOR
  double length = 1000.0;      // TAMANHO FÍSICO DO DOMÍNIO (QUADRADO LENGTH X LENGTH)
  double dx = length / (n+1);  // TAMANHO DE CADA CÉLULA (ADICIONA +1 PARA EXCLUIR FRONTEIRAS FIXAS)
  double dt = 0.5 / nsteps;    // INTERVALO DE TEMPO (TEMPO TOTAL 0.5s DIVIDIDO PELOS PASSOS)

  // PARÂMETRO DE ESTABILIDADE R = ALPHA*DT/ (DX^2)
  double r = alpha * dt / (dx * dx);

  // IMPRESSÃO DAS CONFIGURAÇÕES DE EXECUÇÃO
  printf("\n");
  printf(" MMS heat equation\n\n");
  printf(LINE);
  printf("Problem input\n\n");
  printf(" Grid size: %d x %d\n", n, n);
  printf(" Cell width: %E\n", dx);
  printf(" Grid length: %lf x %lf\n", length, length);
  printf("\n");
  printf(" Alpha: %E\n", alpha);
  printf("\n");
  printf(" Steps: %d\n", nsteps);
  printf(" Total time: %E\n", dt*(double)nsteps);
  printf(" Time step: %E\n", dt);
  printf(LINE);

  // VERIFICAÇÃO DE ESTABILIDADE DO ESQUEMA
  printf("Stability\n\n");
  printf(" r value: %lf\n", r);
  if (r > 0.5)
    printf(" Warning: unstable\n"); // AVISO DE INSTABILIDADE SE R > 0.5
  printf(LINE);

  // ALOCAÇÃO DINÂMICA DE MEMÓRIA PARA AS MALHAS U E U_TMP (TAMANHO NXN)
  double *u     = malloc(sizeof(double)*n*n);
  double *u_tmp = malloc(sizeof(double)*n*n);
  double *tmp; // PONTEIRO AUXILIAR PARA TROCA DE MALHAS

  // INICIALIZA A MALHA U COM VALORES DA SOLUÇÃO MMS
  initial_value(n, dx, length, u);
  zero(n, u_tmp); // ZERA A MALHA TEMPORÁRIA

  //
  // USO DE DIRETIVAS OPENMP PARA GERENCIAMENTO DE MEMÓRIA NO DISPOSITIVO
  //

  
  // ALOCA EXPLICITAMENTE MEMÓRIA PARA U E U_TMP NO DISPOSITIVO (GPU)
  #pragma omp target enter data map(alloc: u[0:n*n], u_tmp[0:n*n])
  // COPIA OS DADOS INICIAIS DE U DO HOST PARA O DISPOSITIVO PARA EXECUÇÃO
  #pragma omp target update to(u[0:n*n]) 

  // INÍCIO DA CONTAGEM DE TEMPO DO PROCESSO DE SOLUÇÃO
  double tic = omp_get_wtime();

  for (int t = 0; t < nsteps; ++t) {

    // CHAMA A FUNÇÃO SOLVE PARA CALCULAR O PRÓXIMO PASSO TEMPORAL NA GPU
    solve(n, alpha, dx, dt, u, u_tmp);

    // TROCA OS PONTEIROS DAS MALHAS PARA AVANÇAR O TEMPO
    tmp = u;
    u = u_tmp;
    u_tmp = tmp;
  }

  // FIM DA CONTAGEM DE TEMPO DO PROCESSO DE SOLUÇÃO
  double toc = omp_get_wtime();

  // ATUALIZA OS DADOS DA MALHA U DO DISPOSITIVO PARA O HOST APÓS A EXECUÇÃO
  #pragma omp target update from(u[0:n*n]) 
  // LIBERA A MEMÓRIA ALOCADA NO DISPOSITIVO PARA U E U_TMP, EVITANDO VAZAMENTOS
  #pragma omp target exit data map(delete: u[0:n*n], u_tmp[0:n*n])


  //
  // CÁLCULO DA NORMA L2 DO ERRO ENTRE SOLUÇÃO COMPUTADA E SOLUÇÃO ANALÍTICA
  //
  double norm = l2norm(n, u, nsteps, dt, alpha, dx, length);

  // FINALIZAÇÃO DA CONTAGEM DE TEMPO TOTAL DO PROGRAMA
  double stop = omp_get_wtime();

  // IMPRESSÃO DOS RESULTADOS
  printf("Results\n\n");
  printf("Error (L2norm): %E\n", norm);
  printf("Solve time (s): %lf\n", toc-tic);
  printf("Total time (s): %lf\n", stop-start);
  printf(LINE);

  // LIBERAÇÃO DA MEMÓRIA ALOCADA NO HOST
  free(u);
  free(u_tmp);

}


// FUNÇÃO PARA DEFINIR OS VALORES INICIAIS NA MALHA SEGUNDO A SOLUÇÃO MMS
void initial_value(const int n, const double dx, const double length, double * restrict u) {

  double y = dx;
  for (int j = 0; j < n; ++j) {
    double x = dx; // POSIÇÃO FÍSICA X
    for (int i = 0; i < n; ++i) {
      u[i+j*n] = sin(PI * x / length) * sin(PI * y / length); // ATRIBUI FUNÇÃO SENOIDAL
      x += dx;
    }
    y += dx; // POSIÇÃO FÍSICA Y
  }

}


// FUNÇÃO PARA ZERAR TODOS OS ELEMENTOS DO VETOR U
void zero(const int n, double * restrict u) {

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      u[i+j*n] = 0.0;
    }
  }

}


// FUNÇÃO QUE EXECUTA O PASSO DE TEMPO EXPLÍCITO DA EQUAÇÃO DO CALOR
void solve(const int n, const double alpha, const double dx, const double dt, const double * restrict u, double * restrict u_tmp) {

    const double r = alpha * dt / (dx * dx); // PARÂMETRO DE ESTABILIDADE
    const double r2 = 1.0 - 4.0 * r;        // COEFICIENTE CENTRAL NO ESQUEMA EXPLÍCITO

    // DIRETIVA OPENMP PARA EXECUTAR O LOOP NESTE NÍVEL EM PARALELO NA GPU
    #pragma omp target teams distribute parallel for collapse(2) \
        map(to: u[0:n*n]) map(from: u_tmp[0:n*n]) \
        firstprivate(n, r, r2)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            // CÁLCULO DO VALOR DA MALHA NA PRÓXIMA ITERAÇÃO TEMPORAL
            u_tmp[i + j * n] = r2 * u[i + j * n]
                             + r * ((i < n - 1) ? u[i + 1 + j * n] : 0.0)    // VALOR AO LADO DIREITO, SE NÃO FRONTEIRA
                             + r * ((i > 0)     ? u[i - 1 + j * n] : 0.0)    // VALOR AO LADO ESQUERDO, SE NÃO FRONTEIRA
                             + r * ((j < n - 1) ? u[i + (j + 1) * n] : 0.0)  // VALOR ACIMA, SE NÃO FRONTEIRA
                             + r * ((j > 0)     ? u[i + (j - 1) * n] : 0.0); // VALOR ABAIXO, SE NÃO FRONTEIRA
        }
    }
}


// FUNÇÃO QUE RETORNA A SOLUÇÃO ANALÍTICA PARA UM PONTO (X,Y) E TEMPO T
double solution(const double t, const double x, const double y, const double alpha, const double length) {

  return exp(-2.0*alpha*PI*PI*t/(length*length)) * sin(PI*x/length) * sin(PI*y/length);

}


// FUNÇÃO QUE CALCULA A NORMA L2 DO ERRO ENTRE A SOLUÇÃO COMPUTADA E A SOLUÇÃO ANALÍTICA
double l2norm(const int n, const double * restrict u, const int nsteps, const double dt, const double alpha, const double dx, const double length) {

    double time = dt * (double)nsteps; // TEMPO FINAL DA SIMULAÇÃO
    double norm = 0.0;                 // VARIÁVEL PARA ACUMULAR O SOMATÓRIO

    // DIRETIVA OPENMP PARA PARALELIZAR O CÁLCULO DO SOMATÓRIO NA GPU
    #pragma omp target teams distribute parallel for collapse(2) reduction(+:norm) \
        map(to: u[0:n*n]) firstprivate(n, dx, length, alpha, time)
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            double x = dx * (i + 1);
            double y = dx * (j + 1);
            double exact = solution(time, x, y, alpha, length); // VALOR EXATO DA SOLUÇÃO
            double diff = u[i + j * n] - exact;                 // DIFERENÇA ENTRE SOLUÇÃO NUMÉRICA E ANALÍTICA
            norm += diff * diff;                                // ACUMULA O QUADRADO DA DIFERENÇA
        }
    }

    return sqrt(norm); // RETORNA A RAIZ QUADRADA DO SOMATÓRIO (NORMA L2)
}
