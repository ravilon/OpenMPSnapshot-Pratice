#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>

//---------------------------------------VARIÁVEIS GLOBAIS
int N_THREADS = 0;
int CACHE_SIZE = 0;
int CHUNCK_SIZE = 0;
double seq_time = 0;
double par_time = 0;



//---------------------------------------FUNÇÕES UTILITÁRIAS
//FUNÇÃO PARA O CÁCULO DO TEMPO
double timestamp(void){
	struct timeval tp;
	gettimeofday(&tp,NULL);
	return ((double)(tp.tv_sec + (double)tp.tv_usec/1000000)); //em segundos
}



// FUNÇÃO UTILITÁRIA, que vai retornar o máximo entre dois inteiros
int max(int a, int b){
	return (a > b) ? a : b;
}


//---------------------------------------FUNÇÃO knapSack
int knapSack(int W, int wt[], int val[], int n){
	double s_init, s_end, p_init, p_end = 0;
	s_init = timestamp();

    //aloca o vetor com uma posição a mais que o peso da mochila e inicializa com zeros
	int *dp = (int*) calloc(W+1, sizeof(int));	//armazenará o vetor de solução final
	int *aux = (int*) calloc(W+1, sizeof(int)); //utilizado para guardar os itens que devem ser comparados na próxima iteração
	int j, weight, calc, value, max_v = 0;

	//define o tamanho do chunk a ser utilizado. 
	//estamos trabalhando com dois vetores de inteiros (aux,dp) e dois valores inteiros (weight, value) 
	//que seria muito bom guarda-los em memória e não fossem despejados quando uma nova linha de cache chegasse
	CHUNCK_SIZE = CACHE_SIZE / abs( (N_THREADS * sizeof(int)) );

	int i = 1;
	int w = W;
	
	s_end = timestamp();
	seq_time += s_end - s_init;

	for (i = 0; i < n; i++){
		weight = wt[i]; //armazena em uma única variável, o que pode permitir que esse dado não sofra despejo caso novas linhas precisem ser instaladas na cache.
		value = val[i];
		
		#pragma omp parallel num_threads(N_THREADS) shared(dp, weight, value, W) private(w) firstprivate(i) proc_bind(close)
		{
		#pragma opm single
			p_init = timestamp();
		#pragma omp for schedule(guided,CHUNCK_SIZE)
		for (w = W; w >= 0; w--){
			if (weight <= w)
			//encontra o maior valor utilizando o dp da iteração anterior e armazena o resultado em um novo vetor
			//fundamental visto que outra thread poderia alterar o dp
			//precisamos garantir que sempre o valor comparado por uma nova iteração sejam os valores da iteração antiga
			aux[w] = max(dp[w],
						 dp[w - weight] + value);
		}

		#pragma omp barrier
		#pragma opm single
		{
			p_end = timestamp();
			par_time += p_end - p_init;
		}
    	} //FIM #pragma omp parallel

	//TEMPO SEQUENCIAL RELEVANTE
	s_init = timestamp();
	memmove(dp, aux, (W+1) * sizeof(int) );
	s_end = timestamp();
	seq_time += s_end - s_init;
	}
    return dp[W]; //retorna o maior valor encontrado
}


//---------------------------------------MAIN
int main(int argc, char **argv){
	if(argc != 3){
		printf("argc = %d\n", argc);
		// printf("argv[0] = %s\n", argv[0]);
		// printf("argv[1] = %s\n", argv[1]);
		// printf("argv[2] = %s\n", argv[2]);
		// printf("argv[3] = %s\n", argv[3]);
		// printf("argv[4] = %s\n", argv[4]);

		printf("USAGE: p_knapsack <THREADS> <CACHE_SIZE> <( < file_of_items )> \n");
		return -1;
	}

	//---------LEITURA DA ENTRADA
	int n, W;

	N_THREADS = atoi(argv[1]);
	// printf("N_THREADS = %d\n", N_THREADS);

    CACHE_SIZE = atoi(argv[2]);
    CACHE_SIZE *= 1024;
	// printf("CACHE_SIZE = %d\n", CACHE_SIZE);


	scanf("%d %d", &n, &W);
	int *val = (int*) calloc(n, sizeof(int)); //VALOR(profit)
	int *wt = (int*) calloc(n, sizeof(int)); //PESOS(weight)

	int i;
	for (i = 0; i < n; ++i){
		scanf("%d %d", &(val[i]), &(wt[i])); 
	}
	//--------------------------


	//---------ALGORITMO
    int max_value = knapSack(W, wt, val, n);
	//--------------------------

	printf("%d;%d;%d;%d;%g;%g;%g\n", N_THREADS,n,W,max_value,seq_time+par_time,seq_time,par_time); //PRINT FINAL

	free(val);
	free(wt);
	return 0;
}