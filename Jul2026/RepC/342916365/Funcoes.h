/* --- Funções ---*/

//define quantidade de nucleos
void DefineNucleos(){

//configuração para informações do sistema
#ifdef _WIN32 
#ifndef _SC_NPROCESSORS_ONLN
SYSTEM_INFO info;
GetSystemInfo(&info);
#define sysconf(a) info.dwNumberOfProcessors
#define _SC_NPROCESSORS_ONLN
#endif
#endif

//numero de nucleos disponíveis
nprocs = sysconf(_SC_NPROCESSORS_ONLN); 

}	

//inicia a fila
void IniciaFila(struct Fila *fila){
fila->frente = 0;
fila->tras = 0;
strcpy(fila->linha[fila->frente].conteudo, "");
}

//enfileira
void Enfileira(struct Fila *fila){

//variável que representa uma linha
struct Linha l;

//Leitura do arquivo
FILE *pont_arq;
char texto_str[tamLinha];

//abrindo o arquivo_frase em modo "somente leitura"
pont_arq = fopen("./Arquivo/arquivo.txt", "r");

//enquanto não for fim de arquivo o looping será executado e será impresso o texto
while(fgets(texto_str, tamLinha, pont_arq) != NULL){

//adiciona valores a struct linha
strcpy(l.conteudo, texto_str);

//enfileira linha
strcpy(fila->linha[fila->tras].conteudo, l.conteudo);
fila->tras=fila->tras+1;	
}
}

//desenfileira uma posicao da fila
int Desenfileira(struct Fila *fila){

//variaveis de interação
int i=0, j=0, k=0;

//string que representa uma linha
char l[tamLinha] = "";

//zona crítica executada pelas threads uma por vez
#pragma omp critical 
{
//se ainda existir elementos na fila
if (fila->frente < fila->tras){

//transforma linha em maiúsculas
for (i=0;i<strlen(fila->linha[fila->frente].conteudo);i++){
l[i] = toupper (fila->linha[fila->frente].conteudo[i]);	
}

//executa função de contagem de ocorrencias
Substring_count(l, palavra);
}

//retira linha
strcpy(fila->linha[fila->frente].conteudo, " ");

//desenfileira uma posicao
fila->frente += 1;
}

//verifica se já não percorreu toda a fila
if (fila->frente > fila->tras){
return 0;
}

return 1;
}

//função que faz a contagem de ocorrencias de substring em string
void Substring_count(char* string, char* substring) {
int i, j, l1, l2;
int found = 0;

//tamanho das strings	
l1 = strlen(string);
l2 = strlen(substring);

for(i = 0; i < l1; i++) {
found = 1;
for(j = 0; j < l2; j++) {
if(string[i+j] != substring[j]) {
found = 0;
break;
}
}

if(found == 1) {
if (i == 0 && i != (l1-l2)){
if (string[i+l2] == ' ' || string[i+l2] == '?'|| string[i+l2] == '!' || string[i+l2] == '.' || string[i+l2] == ','){
contador++;
i = i + l2 -1;	
}		
}
else if (string[i-1] == ' ' && (string[i+l2] == ' ' || string[i+l2] == '?'|| string[i+l2] == '!' || string[i+l2] == '.' || string[i+l2] == ',')){
contador++;
i = i + l2 -1;	

}
else if (i == (l1-l2-1)){
if (string[i-1] == ' '){
contador++;
i = i + l2 -1;	
}
}
else if (i == 0 && i == (l1-l2)){
contador++;
i = i + l2 - 1;
}
else if ( i == l1-l2){
if (string[i-1] == ' '){
contador++;
i = i+l2-1;
}
}
}			
}

}

//Função teste para exibir a fila
void ExibeFila(struct Fila *fila){
int i;
for (i=fila->frente;i<fila->tras;i++){
printf ("%s", fila->linha[i].conteudo);
}

}

//transforma palavra pesquisada em uppercase
void TransformaPalavra(char palavraPesquisada[]){
//strcpy de palavra pesquisada
strcpy(palavra, palavraPesquisada);

int i; 

//converte palavra para maiúscula
for (i=0;i<strlen(palavra);i++){
palavra[i] = toupper(palavra[i]);
}
}

//exibe ocorrencias de palavra no texto
void ExibeOcorrencia(){

printf ("----------- INFORMACOES DO SISTEMA -------------\n\n");
printf ("       %d NUCLEOS DO PROCESSADOR EM USO\n\n\n", nprocs);

printf ("----------------- OCORRENCIAS ------------------\n\n");
printf (" %d OCORRENCIA(S) DA PALAVRA '%s' NO TEXTO\n", contador, palavra);
}

