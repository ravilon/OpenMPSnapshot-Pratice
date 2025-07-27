#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "structs.h"
#include "utils.h"

short cost(int x)
{
    int i, n_iter = 20;
    double dcost = 0;
    for(i = 0; i < n_iter; i++)
		dcost += pow(sin((double) x),2) + pow(cos((double) x),2);
    return (short) (dcost / n_iter + 0.1);
}

/*

 c[i,j]

 0							if i=0 or j=0
 c[i-1,j-1]+1				if i,j>0 and xi=yj
 max(c[i,j-1], c[i-1,j])	if i,j>0 and xi!=yj
 */

/*--------------------------------------------------
Função: funct_parallel
Objectivo: percorrer toda a diagonal em paralelo de modo a completar a matriz da forma mais eficiente possivel
Devido a dependencias de dados, cada diagonal da matriz tem de ser feita em série, pois uma diagonal é sempre dependente
da anterior. No entanto, quando a completar cada diagonal, ja se pode executar em paralelo, pois cada valor de uma diagonal é
independente dos outros.
Variaveis:
matrix -> variavel auxiliar que aponta para a matriz de dados criada em readFile
control -> a trabalhar em sincronia com as threads e o for, control percorre a diagonal de 0 a counter(tamanho da diagonal actual)
ii, jj -> variavies auxiliares que se asseguram de que i e j sao dependentes de control
---------------------------------------------------*/
 
void funct_parallel(LcsMatrix *lcsMtx, int i, int j, int counter) {

	int **matrix = lcsMtx->mtx;
	int control;
	int ii=i,jj=j;

#pragma omp for schedule(dynamic) private(control)  //percorrer a diagonal em paralelo. Cada thread tem no seu stack um valor de control
	for(control = 0; control < counter; control++) {
		i = ii + control;
		j = jj - control;

		if(lcsMtx->seqLine[i] == lcsMtx->seqColumn[j]) {	//caso os valores sejam iguais, incrementa o dado do ponto i-1, j-1 e guarda
			matrix[i][j] = matrix[i-1][j-1] + cost(control);
                    
		} else {
			matrix[i][j] = matrix[i][j-1] >= matrix[i-1][j] ? matrix[i][j-1] : matrix[i-1][j];	//caso sejam diferentes, ve o maior de dois elementos e guarda
		}
					
	}

	return;
}
/*----------------------------------------
Função: fillMatrix
Objectivo: Completar a matriz com os dados necessários para, a seguir, obter a sequencia similar das duas strings
Variaveis:
maxDiag -> contem o tamanho maximo que a diagonal pode ter
maxAbs, incCounterVertical, aux -> variaveis auxiliares usadas para controlar o counter
Counter -> antes de entrar na função funct_parallel, contem o numero de elementos que serão lidos na diagonal

Nota: nesta função todas as threads estão a executar o mesmo codigo, sendo o "verdadeiro paralelismo" executado na funçao
funct_parallel. Apesar do codigo desnecessário a ser corrido, é mais eficiente do que criar e remover threads sempre que se entrava
na função funct_parallel
-----------------------------------------*/

void fillMatrix(LcsMatrix *lcsMtx) {

	int i=1;
	int j, maxDiag, maxAbs;
	int counter=1;
	int incCounterVertical=1;
	int aux;
	
	/*maxDiag vai conter o minimo dos dois, pois é a a dimensao max de uma diagonal,
	maxAbs contem o maximo dos dois, para ajuda no calculo do counter caso nºcolunas < nºlinhas*/
	maxDiag = (lcsMtx->cols >= lcsMtx->lines) ? lcsMtx->lines : lcsMtx->cols;
	maxAbs = (lcsMtx->cols > lcsMtx->lines) ? lcsMtx->cols : lcsMtx->lines;

#pragma omp parallel private(j, aux) firstprivate(i, counter, incCounterVertical) 
{
	//vamos percorrer toda a 2ªlinha da matriz e, a partir daí, ir de diagonal em diagonal para completar a matriz
	for(j=1; j <= lcsMtx->cols; j++) {

		#pragma omp barrier //todas as threads têm de esperar que a ultima diagonal esteja completa antes de passarem para a proxima
		
		funct_parallel(lcsMtx, i, j, counter);
		
		counter++;	//cada vez que nos deslocamos mais para a direita o numero de valores na diagonal aumenta. So nao pode exceder maxDiag

		if(counter > maxDiag)
			counter=maxDiag;
	}
	
	j=lcsMtx->cols;
	aux = j;
	
#pragma omp barrier //antes de começar a percorrer a ultima coluna, é preciso assegurar que todas as threads anteriores acabaram

	// Percorrer toda a ultima coluna da matriz para ober os valores abaixo da diagonal principal da matriz
	for(i=2; i <= lcsMtx->lines; i++) {

		if (aux == maxAbs) {	//calculo auxiliares que asseguram o valor correcto de counter caso a matriz tenha nºlinhas<nºcolunas ou ao contrario
			incCounterVertical = -1;
		} else {
			aux++;
		}
		
		counter += incCounterVertical;
		if (counter > maxDiag)
			counter = maxDiag;
		
		#pragma omp barrier

		funct_parallel(lcsMtx, i, j, counter);	//tal como no ciclo anterior, ele percorre diagonal a diagonal e completa a matriz
	}
}
//	printLcsMatrix(lcsMtx);
	
	return;
}

/*---------------------------------------------------------
Função: findLongestCommonSubsequence
Objectivo: aplicar o algoritmo, escrito no enunciado, na matriz e obter a sequencia equivalente das duas strings
Variaveis:
i, j -> percorrem a matriz
counter -> +1 sempre que existe igualdade. Guarda o numero de caracteres da string equivalente
lcsStringInverted -> sabendo que a string equivalente nunca podera ser maior que o minimo do numero colunas/linhas, guarda a string
invertida a medida que se percorre a matriz
lcsString -> depois de obtido o tamanho e a matriz invertida, inverte esta ultima e guarda assim a resposta final do problema
----------------------------------------------------------*/

LcsResult findLongestCommonSubsequence(LcsMatrix *lcsMtx) {

	LcsResult result;

	// start at button right of the matrix
	int i = lcsMtx->lines;
	int j = lcsMtx->cols;

	char *lcsStringInverted = (char *)calloc(i<j ? i+1 : j+1, sizeof(char));
	char *lcsString;
	int counter = 0;

	/*aplica o algoritmo tal como dito no enunciado. Começa no valor do canto inferior direito da matriz
	Caso exista equivalencia, desloca-se na diagonal e guarda o caracter. Caso contrário, desloca-se para
	o maior dos dois valores, acima ou a esquerda dele, na matriz*/
	while (i!=0 && j!=0) {

		// check match
		if (lcsMtx->seqLine[i] == lcsMtx->seqColumn[j]) {
			lcsStringInverted[counter] = lcsMtx->seqLine[i];
			counter = counter + 1;

			// move diagonally
			i--;
			j--;

		} else {
			// check which is larger
			if (lcsMtx->mtx[i][j-1] >= lcsMtx->mtx[i-1][j]) {
				// move to largest
				j--;
			} else {
				i--;
			}
		}
	}
	
	//encerra a sequencia final adicionando o \0 no fim
	lcsStringInverted[counter] = '\0';
	
	//inverte a matriz lcsStringInverted e guarda a sequencia final
	lcsString = (char *)calloc(counter, sizeof(char));

	for (i=counter-1; i>=0; i--) {
		lcsString[counter-1-i] = lcsStringInverted[i];
	}
	free(lcsStringInverted);

	result.counter = counter;
	result.sequence = lcsString;

	return result;
}

