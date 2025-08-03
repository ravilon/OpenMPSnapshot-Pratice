/*
* Lucas Bergmann e Ricardo Somariva
* Programação Paralela 2018/2
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <ctype.h>
#include <string.h>
#include <time.h>
#include <omp.h>

/**
* Disponível em: https://www.geeksforgeeks.org/wildcard-character-matching/
*/
// The main function that checks if two given strings 
// match. The first string may contain wildcard characters 
bool match(char *first, char * second) { 
// If we reach at the end of both strings, we are done 
if (*first == '\0' && *second == '\0') 
return true; 

// Make sure that the characters after '*' are present 
// in second string. This function assumes that the first 
// string will not contain two consecutive '*' 
if (*first == '*' && *(first+1) != '\0' && *second == '\0') 
return false; 

// If the first string contains '?', or current characters 
// of both strings match 
if (*first == '?' || *first == *second) 
return match(first+1, second+1); 

// If there is *, then there are two possibilities 
// a) We consider current character of second string 
// b) We ignore current character of second string. 
if (*first == '*') 
return match(first+1, second) || match(first, second+1); 
return false; 
} 

/**
* Disponível em: https://stackoverflow.com/questions/1841758/how-to-remove-punctuation-from-a-string-in-c?noredirect=1&lq=1
*/
void remove_punct_and_make_lower_case(char *p) {
char *src = p, *dst = p;

while (*src) {
if (ispunct((unsigned char)*src)) {
/* Skip this character */
src++;
} else if (isupper((unsigned char)*src)) {
/* Make it lowercase */
*dst++ = tolower((unsigned char)*src);
src++;
} else if (src == dst) {
/* Increment both pointers without copying */
src++;
dst++;
} else {
/* Copy character */
*dst++ = *src++;
}
}

*dst = 0;
}

int main(void) {
/* declare a file pointer */
FILE *infile;
char *buffer;
long numbytes;
char pattern[100];
char *array[400000];

/* open an existing file for reading */
infile = fopen("texto.txt", "r");

/* quit if the file does not exist */
if(infile == NULL)
return 1;

/* Get the number of bytes */
fseek(infile, 0L, SEEK_END);
numbytes = ftell(infile);

/* reset the file position indicator to 
the beginning of the file */
fseek(infile, 0L, SEEK_SET);

/* grab sufficient memory for the 
buffer to hold the text */
buffer = (char*)calloc(numbytes, sizeof(char));	

/* memory error */
if(buffer == NULL)
return 1;

/* copy all the text into the buffer */
fread(buffer, sizeof(char), numbytes, infile);
fclose(infile);

printf("WILDCARD PATTERN MATCHING");
printf("\n\n\nDigite o padrão desejado: ");
scanf("%s", pattern);
printf("\n\n");

char * line = strtok(strdup(buffer), "\n");
int numLines = 0;


// Este loop serve p/ popular o array e contar o numero de linhas do texto p/ utilizar no for paralelizado
while(line != NULL) {
remove_punct_and_make_lower_case(line); 
array[numLines] = line;
line  = strtok(NULL, "\n");
numLines++;
}

// Início da execução
clock_t timeStart = clock(); 

// Pode-se utilizar este comando para setar o numero de threads
// ao invés de variáveis de ambiente
//omp_set_num_threads(4);

printf("LINHAS QUE OCORREU O PADRÃO\n");

int numPatternFound = 0;
char linhaChar[100];
int i = 0;
#pragma omp parallel
{
#pragma omp for reduction(+:numPatternFound)
for(int i = 0; i < numLines; i++) {
if(match(pattern, array[i])) {
printf("%d: %s\n", i, array[i]);
numPatternFound++;
}
}
}

// Fim da execução
clock_t timeFinish = clock();

// Calcula a diferença de tempo
double executionTime = (double)(timeFinish - timeStart) / CLOCKS_PER_SEC;

printf("\n\n");
printf("TEMPO DE EXECUÇÃO: %lf\n", executionTime);
printf("O PADRÃO OCORREU %d VEZES\n", numPatternFound);
printf("\nFIM DA EXECUÇÃO\n");


free(buffer);
return 0;
}