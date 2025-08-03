#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const int MAX_LINES = 1000;
const int MAX_LINE = 80;

void como_usar(char* prog_name);
void pegar_texto(char* lines[], int* line_count_p);
void tokens(char* lines[], int line_count, int thread_count);
char* strtok_manual(char* seps, char** next_string_p);
int separator(char* current, char* seps);

void pegar_texto(char* lines[], int* line_count_p) {
char* line = malloc(MAX_LINE * sizeof(char));
int i = 0;
char* fg_rv;

fg_rv = fgets(line, MAX_LINE, stdin);
while (fg_rv != NULL) {
lines[i++] = line;
line = malloc(MAX_LINE * sizeof(char));
fg_rv = fgets(line, MAX_LINE, stdin);
}
*line_count_p = i;
}

void tokens(char* lines[], int line_count, int thread_count) {
int my_rank, i, j;
char *my_token, *next_string;

#pragma omp parallel num_threads(thread_count) default(none) private(  my_rank, i, j, my_token, next_string) shared(lines, line_count)
{
my_rank = omp_get_thread_num();
#pragma omp for schedule(static, 1)
for (i = 0; i < line_count; i++) {
printf("Thread %d > line %d = %s\n", my_rank, i, lines[i]);
j = 0;
next_string = lines[i];
my_token = strtok_manual(" \t\n", &next_string);
while (my_token != NULL) {
printf("Thread %d > token %d = %s\n", my_rank, j, my_token);
my_token = strtok_manual(" \t\n", &next_string);
j++;
}
if (lines[i] != NULL)
printf("Thread %d > After tokenizing, my line = %s\n", my_rank,
lines[i]);
}
}
}

char* strtok_manual(char* seps, char** next_string_p) {
char* token;
int length = 0;
char* start;
char* current = *next_string_p;

while (separator(current, seps))
if ((*current == '\0') || (*current == '\n'))
return NULL;
else
current++;
start = current;

while (!separator(current, seps)) {
length++;
current++;
}

token = (char*)malloc((length + 1) * sizeof(char));
strncpy(token, start, length);
token[length] = '\0';

*next_string_p = current;

return token;
}

int separator(char* current, char* seps) {
int len = strlen(seps);
int i;

if (*current == '\0') return 1;
for (i = 0; i < len; i++)
if (*current == seps[i]) return 1;
return 0;
}

int main(int argc, char* argv[]) {
int thread_count, i;
char* lines[1000];
int line_count;

thread_count = strtol(argv[1], NULL, 10);

printf("Enter text\n");
pegar_texto(lines, &line_count);
tokens(lines, line_count, thread_count);

for (i = 0; i < line_count; i++)
if (lines[i] != NULL) free(lines[i]);

return 0;
}