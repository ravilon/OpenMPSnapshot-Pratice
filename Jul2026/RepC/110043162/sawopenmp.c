#include<omp.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<limits.h>
#include<float.h>
#define lin 209555
#define col 4
#define NUM_THREADS 8

//Para compilar gcc sawopenmp.c -fopenmp -o sawopenmp

char** str_split(char* a_str)
{

char ** result = NULL;
char *tmp = NULL;
int length = 0;

result = (char**)malloc(sizeof(char*));

tmp = strtok(a_str, ",");
while(tmp != NULL)
{
result[length] = tmp;
length++;
tmp = strtok(NULL, ",");
result = (char**)realloc(result, (length + 1) * sizeof(char*));
}
return result;
}

char* read_line(FILE * f)
{
char c;
char *retval = NULL;
int i = 0;

while(!feof(f) && c != '\n')
{
c = fgetc(f);
retval = (char*)realloc(retval, (i+1) * sizeof(char));
retval[i++] = c;
}
retval = (char*)realloc(retval, (i+1) * sizeof(char));
retval[i] = '\0';
return retval;
}

//fun��o do qsort para ordena��o
int ordena(const void * a,const void * b)
{
if(*(int*)a == *(int*)b)
return 0; //iguais
else if (*(int*)a > *(int*)b)
return -1; // vem antes
else
return 1; // vem depois
}

void CalculavetorMaior(float vetorMaior[lin] , float  matrizDados[lin][col])
{
#pragma omp parallel num_threads(4)
#pragma omp for schedule(static)
for(int j=0; j<col; j++)
{
float maior = matrizDados[0][j];
for(int i=0; i<lin; i++)
{
if(matrizDados[i][j]>maior)   //este trecho recebe o maior valor da linha da matriz
{
maior=matrizDados[i][j];  //e armazena este valor em um vetor
}
}
vetorMaior[j]=maior;
maior=0;
}
}

void CalculavetorMenor(float vetorMenor[lin] , float  matrizDados[lin][col])
{
#pragma omp parallel num_threads(4)
#pragma omp for schedule(static)
for(int j=0; j<col; j++)
{
float menor = matrizDados[0][j];
for(int i=0; i<lin; i++)
{
if(menor > matrizDados[i][j])
{
menor = matrizDados[i][j];
}
}
vetorMenor[j]=menor;
menor=0;
}
}

//fun��o para normalizar a matriz de dados onde op=0 -> max e op=1 -> min
void normaliza(float matrizDadosNormalizada[lin][col], float matrizDados[lin][col], float vetorMaior[lin], float vetorMenor[lin], int op)
{
#pragma omp parallel
#pragma omp for schedule(dynamic)
for(int i=0; i<lin; i++)
{
for(int j=0; j<7; j++)
{
if(op == 0)
{
matrizDadosNormalizada[i][j] = ((matrizDados[i][j] - vetorMenor[j]) / (vetorMaior[j] - vetorMenor[j]));
}
else if(op == 1)
{
matrizDadosNormalizada[i][j] = ((vetorMaior[j] - matrizDados[i][j]) / (vetorMaior[j] - vetorMenor[j]));
}
}
}
}

void geraRanking(float matrizDadosNormalizada[lin][col], float vetorResul[lin], float vetorPesos[lin])
{
int i,j;

//Multiplica��o da Matriz de dados pelo Vetor de pesos colocando em "vetorResul"
for(i=0; i<lin; i++)
{
vetorResul[i] = 0;
for(j=0; j<7; j++)
{
vetorResul[i] += matrizDadosNormalizada[i][j] * vetorPesos[j];
}
}
}

void exibivetorMaior(float vetorMaior[col])
{

int i;
printf("Maior elemento de cada coluna!\n"); //exibe o vetor com o maior elemento de cada linha
for(i=0; i<col; i++)
{
printf("%.3f \n",vetorMaior[i]);
}
}

void exibivetorMenor(float vetorMenor[col])
{

int i;
printf("Menor elemento de cada coluna!\n"); //exibe o vetor com o maior elemento de cada linha
for(i=0; i<col; i++)
{
printf("%.3f \n",vetorMenor[i]);
}
}

void exibivetorResul(float vetorResul[lin])
{
int i;
printf("Impress�o do produto Matriz x Vetor de Pesos\n");
for(i=0; i<lin; i++)
{
printf("Elemento[%d] = %.3f \n", i, vetorResul[i]);
}
}

void exibiMatriz(float matrizDados[lin][col])
{
//imprime matriz normal
int i,j;
printf("\n Matriz normal\n");
for(i=0; i<lin; i++)
{
for(j=0; j<7; j++)
{
printf("[%.2f]", matrizDados[i][j]);
}
printf("\n");
}
}

void exibiMatrizNormalizada(float matrizDadosNormalizada[lin][col])
{
//imprime matriz normalizada
int i,j;
printf("Matriz normalizada\n");
for(i=0; i<lin; i++)
{
for(j=0; j<col; j++)
{
printf("[%.3f]", matrizDadosNormalizada[i][j]);
}
printf("\n");
}
}

int main (void )
{
//int i, j;

//ponteiro para arquivo
FILE *f;

//função para realizar a abertura do arquivo
f = fopen("./day_city.csv","r");

float matrizDados[lin][col];

// if de controle para ver se o arquivo esta disponivel, caso não fecha
if (f == NULL)
{
printf("Erro na Abertura do Arquivo\n");
system("pause");
exit (1);
}
else
{
int index = 0;
//read_line(f);
//laço para fazer a leitura do arquivo
while(!feof(f))
{
char **tokens = NULL;
char *s = read_line(f);

if(strlen(s) > 1)
{
tokens = str_split(s);
matrizDados[index][0] = atof(tokens[0]); //-> sensor.temp_day
matrizDados[index][1] = atof(tokens[1]); //-> sensor.humidity
matrizDados[index][2] = atof(tokens[2]); //-> sensor.clouds
matrizDados[index][3] = atof(tokens[3]); //-> sensor.speed

//Comenta dps os printf quando for testar o código de fato
//printf("temp_day = %lf; ", matrizDados[index][1]);
//printf("humidity = %lf; ", matrizDados[index][2]);
//printf("clouds = %lf; ", matrizDados[index][3]);
//printf("speed = %lf\n", matrizDados[index][4]);
}
index++;
}
}
//printf("Finalizada Matriz\n");
//return 0;

//Declaracao da Matriz e vetores
float vetorPesos[col]     =  {0.162, 0.162, 0.162, 0.162};

float vetorMaior[col] = {};
float vetorMenor[col] = {};
float vetorResul[col] = {};

int op = 0;
int i,j;
float matrizDadosNormalizada[lin][col] = {};
//float aux1[lin] = {};
omp_set_num_threads(NUM_THREADS);


#pragma omp for schedule(static,8)                
for(int j=0; j<col; j++)
{
float maior = 0;
for(int i=0; i<lin; i++)
{
if(matrizDados[i][j]>maior)   //este trecho recebe o maior valor da linha da matriz
{
maior=matrizDados[i][j];  //e armazena este valor em um vetor
}
}
//printf("%f\n", maior);
vetorMaior[j]=maior;
}


#pragma omp for schedule(static,8)                
for(int j=0; j<col; j++)
{
float menor = FLT_MAX;
for(int i=0; i<lin; i++)
{
if(menor > matrizDados[i][j])
{
menor=matrizDados[i][j];
//printf("%f\n", menor);
}
}
//printf("%lf", menor);
vetorMenor[j]=menor;
}

#pragma omp barrier

//tem que colocar um lock aqui pra n�o fazer a normaliza��o enquanto
// se encontra os maiores e menores dados da matriz que est�o sendo salvos nos vetores

#pragma omp for schedule(static,8)
for(int i=0; i<lin; i++)
{
for(int j=0; j<col; j++)
{
op=1;
if(op == 0)
{
matrizDadosNormalizada[i][j] = ((matrizDados[i][j] - vetorMenor[j]) / (vetorMaior[j] - vetorMenor[j]));
}
else if(op == 1)
{
matrizDadosNormalizada[i][j] = ((vetorMaior[j] - matrizDados[i][j]) / (vetorMaior[j] - vetorMenor[j]));
}
}
}

//float sum = 0;
// faz a multiplicasao por pesos em paralelo
#pragma omp for
for(i=0; i<lin; i++)
{
vetorResul[i] = 0;
for(j=0; j<col; j++)
{
vetorResul[i] += matrizDadosNormalizada[i][j] * vetorPesos[j];
//sum += vetorResul[i] ;
}
}



// qsort(vetorResul,lin,sizeof(float), ordena);

//exibiMatriz(matrizDados);
//exibivetorMaior(vetorMaior);
//exibivetorMenor(vetorMenor);
// exibiMatrizNormalizada(matrizDadosNormalizada);
exibivetorResul(vetorResul);

return 0;
}