#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <mpi.h>
#include <time.h>
#include <omp.h>
#include <math.h>

//#define DEBUG


#define NUMBER_OF_LETTERS 26
#define TABLE_SIZE 26 * 26
// Assumption: seq(i) will always have smaller len than seq(1)
#define SEQ_1_MAX_LEN 3000
#define SEQ_2_MAX_LEN 2000
#define MASTER 0
#define INITSCORE -1000

typedef struct
{
int score;
int offset;
int k;
int seq_len;
char seq[SEQ_2_MAX_LEN];
} Seq_Info;

typedef struct
{
int num_of_seqs;
int seq1_len;
int score_table[NUMBER_OF_LETTERS][NUMBER_OF_LETTERS];
char seq1[SEQ_1_MAX_LEN];
} init_variables;

char *seq1;
init_variables init_pack;
Seq_Info *master_seq_arr;
Seq_Info *worker_seq_arr;
int lineno = 1;
int start_end[2];
int chunk_size;

// Functions:
void MS(char *seq, int k);
void fill_score_mat();
void read_score_table(int argc, char **argv);
void print_score_table();
int load_score_table_from_text_file(const char *fileName);
void skip_white_space();
int read_input_seq();
char *handle_string_from_input();
void toUpperCase(char *str);
void find_score_offset_MS(int index, char* temp);
int check_score(char *seq, int offset);
void Work();
void master_Work(int reminder);
void init(int argc, char **argv, int myRank, int nprocs);
void exit_safely(int myRank);
void get_MPI_init_Datatype(MPI_Datatype *custom_type);
void get_MPI_Seq_Info_Datatype(MPI_Datatype *custom_type);
Seq_Info getIndexOfBestScoreAfterCuda(int* max_arr_after_cuda , int size);

//Cuda functions:
extern int computeOnGPU(char* seq1, char* seq2 ,int seq1_length,int seq2_length, int score_table[][26],int table_size,int* max_arr);

int main(int argc, char **argv)
{
int myRank;
int nprocs;
/* Start up MPI */
MPI_Init(&argc, &argv);
/* Find number of processes */
MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
/* Find process myRank */
MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

init(argc, argv, myRank, nprocs);

MPI_Datatype mpi_seq_info_type;
get_MPI_Seq_Info_Datatype(&mpi_seq_info_type);

int elements_per_proc[nprocs-1];
int displacements[nprocs-1];
int reminder = init_pack.num_of_seqs % (nprocs - 1);

for (int i = 0; i < nprocs; i++)
{
if (i == MASTER)
{
elements_per_proc[i] = init_pack.num_of_seqs;
displacements[i] = 0;
}
else
{
elements_per_proc[i] = (init_pack.num_of_seqs / (nprocs - 1));
displacements[i] = ((i - 1) * (init_pack.num_of_seqs / (nprocs - 1)));
}
}

double start_time;

if (myRank == MASTER)
{
start_time = MPI_Wtime();
if(reminder)
{
//use Cuda on the reminder seqs
master_Work(reminder);
}
MPI_Gatherv(NULL, 0, mpi_seq_info_type, master_seq_arr, elements_per_proc, displacements, mpi_seq_info_type, MASTER, MPI_COMM_WORLD);
}else{ // WORKER PART
Work();
MPI_Gatherv(worker_seq_arr, chunk_size, mpi_seq_info_type, NULL, NULL, NULL, mpi_seq_info_type, MASTER, MPI_COMM_WORLD);
}
//Print the answer
if(myRank == MASTER)
{
for (int i = 0; i < chunk_size; i++){
if(i == init_pack.num_of_seqs - reminder)
printf("REMINDERS:\n");
#ifndef DEBUG
printf("%d. Seq = %s -> Highest alignment score = %d, Offset = %d, K = %d \n", i + 1, master_seq_arr[i].seq, master_seq_arr[i].score, master_seq_arr[i].offset, master_seq_arr[i].k);
#else
printf("%d.  -> Highest alignment score = %d, Offset = %d, K = %d \n", i + 1, master_seq_arr[i].score, master_seq_arr[i].offset, master_seq_arr[i].k);
#endif        
}
printf("Runtime: %lf\n", MPI_Wtime() - start_time);
}
exit_safely(myRank);

MPI_Type_free(&mpi_seq_info_type);

MPI_Finalize();
return EXIT_SUCCESS;
}


/*-----------------------------------------------------------------------
@brief ~~> void Work
The Function will use CUDA on the reminder if there is any.
create a temporary array of Seq_Info, to handle the remaining seqs
and appling it to the reminder part of the master array.


@return: void
-----------------------------------------------------------------------*/
void master_Work(int reminder)
{
int seqOfMasterIndex;
for(int i = 0 ; i<reminder ; i++){
seqOfMasterIndex = init_pack.num_of_seqs - reminder + i;
int resultArrSize = 2*(init_pack.seq1_len - master_seq_arr[seqOfMasterIndex].seq_len+1);
int* max_arr = (int*)malloc(resultArrSize *sizeof(int));
if(max_arr == NULL)
{
fprintf(stderr, "Memory allocation failed.\n");
exit(1);
}
computeOnGPU(init_pack.seq1, master_seq_arr[seqOfMasterIndex].seq, init_pack.seq1_len, master_seq_arr[seqOfMasterIndex].seq_len, init_pack.score_table, NUMBER_OF_LETTERS,max_arr);
Seq_Info temp_seqInfo = getIndexOfBestScoreAfterCuda(max_arr,resultArrSize);

master_seq_arr[seqOfMasterIndex].k = temp_seqInfo.k;
master_seq_arr[seqOfMasterIndex].offset = temp_seqInfo.offset;
master_seq_arr[seqOfMasterIndex].score = temp_seqInfo.score;

free(max_arr);
}
}


Seq_Info getIndexOfBestScoreAfterCuda(int* max_arr_after_cuda , int size){

int tid;
#pragma omp declare reduction(max_score_reduction:Seq_Info :  omp_out = (omp_out.score > omp_in.score ? omp_out : omp_in)) initializer(omp_priv = omp_orig)

Seq_Info temp;
temp.score = -1000;
temp.offset = max_arr_after_cuda[0];
temp.k = max_arr_after_cuda[1];

#pragma omp parallel for reduction(max_score_reduction:temp)
for(int i = 0 ; i < size ; i = i+2)
{
if(max_arr_after_cuda[i] > temp.score){
tid = omp_get_thread_num();
temp.score = max_arr_after_cuda[i];
temp.k = max_arr_after_cuda[i+1];
temp.offset = i/2;
#ifdef DEBUG
printf("[%d] ~~> update best: -score- = %d \n", tid , temp.score);
#endif
}
}
return temp;
}


void init(int argc, char **argv, int myRank, int nprocs)
{
MPI_Datatype mpi_init_datatype;
get_MPI_init_Datatype(&mpi_init_datatype);
MPI_Datatype mpi_seq_info_datatype;
get_MPI_Seq_Info_Datatype(&mpi_seq_info_datatype);

if (myRank == MASTER)
{
read_score_table(argc, argv);
read_input_seq();
}
MPI_Bcast(&init_pack, 1, mpi_init_datatype, MASTER, MPI_COMM_WORLD);

#ifdef DEBUG
printf("~~>[%d] num of seqs = %d, seq1 len= %d seq1 = %s\n", myRank, init_pack.num_of_seqs, init_pack.seq1_len, init_pack.seq1);
print_score_table();
#endif

chunk_size = 0;
int elements_per_proc[nprocs];
int displacements[nprocs];

for (int i = 0; i < nprocs; i++)
{
if (i == MASTER)
{
elements_per_proc[i] = init_pack.num_of_seqs;
displacements[i] = 0;
}
else
{
elements_per_proc[i] = (init_pack.num_of_seqs / (nprocs - 1));
displacements[i] = ((i - 1) * (init_pack.num_of_seqs / (nprocs - 1)));
}
#ifdef DEBUG
printf("~~%d. num_elements=%d, displacement=%d\n", i, elements_per_proc[i], displacements[i]);
#endif
}
if (myRank == MASTER)
{
chunk_size = init_pack.num_of_seqs;
start_end[0] = 0;
start_end[1] = init_pack.num_of_seqs - 1;
MPI_Scatterv(master_seq_arr, elements_per_proc, displacements, mpi_seq_info_datatype, MPI_IN_PLACE, 0, mpi_seq_info_datatype, MASTER, MPI_COMM_WORLD);
}
else
{
chunk_size = (init_pack.num_of_seqs / (nprocs - 1));
start_end[0] = (myRank - 1) * (init_pack.num_of_seqs / (nprocs - 1)) + 1;
start_end[1] = start_end[0] + chunk_size - 1;

worker_seq_arr = (Seq_Info *)malloc(chunk_size * sizeof(Seq_Info));
if (worker_seq_arr == NULL)
{
fprintf(stderr, "Memory allocation failed.\n");
exit(1);
}
MPI_Scatterv(NULL, NULL, NULL, mpi_seq_info_datatype, worker_seq_arr, elements_per_proc[myRank], mpi_seq_info_datatype, MASTER, MPI_COMM_WORLD);
#ifdef DEBUG
for (int i = 0; i < chunk_size; i++)
printf("~~>[%d] seq[%d] = %s\n", myRank, start_end[i], seq_arr[i].seq);
#endif
}
MPI_Type_free(&mpi_seq_info_datatype);
MPI_Type_free(&mpi_init_datatype);
}

/*-----------------------------------------------------------------------
@brief ~~> void Work()
The Function will run the function "find_score_offset_MS" for each seq.


@return: void
-----------------------------------------------------------------------*/
void Work()
{
char temp[SEQ_2_MAX_LEN];
#pragma omp parallel for default(none) firstprivate(temp) shared(chunk_size,worker_seq_arr)
for (int i = 0; i < chunk_size; i++)
{
strcpy(temp, worker_seq_arr[i].seq);
find_score_offset_MS(i, temp);
}  
}

void get_MPI_init_Datatype(MPI_Datatype *custom_type)
{
int block_lengths[4] = {1, 1, TABLE_SIZE, SEQ_1_MAX_LEN};
MPI_Aint displacements[4];
MPI_Datatype types[4] = {MPI_INT, MPI_INT, MPI_INT, MPI_CHAR};
displacements[0] = 0;
displacements[1] = sizeof(int);
displacements[2] = 2 * sizeof(int);
displacements[3] = (2 + TABLE_SIZE) * sizeof(int);
MPI_Type_create_struct(4, block_lengths, displacements, types, custom_type);
MPI_Type_commit(custom_type);
}

void get_MPI_Seq_Info_Datatype(MPI_Datatype *custom_type)
{
int blocklengths[5] = {1, 1, 1, 1, SEQ_2_MAX_LEN};
MPI_Aint displacements[5];
MPI_Datatype types[5] = {MPI_INT, MPI_INT, MPI_INT, MPI_INT, MPI_CHAR};
for (int i = 0; i < 5; i++)
displacements[i] = i * sizeof(int);
MPI_Type_create_struct(5, blocklengths, displacements, types, custom_type);
MPI_Type_commit(custom_type);
}

/*-----------------------------------------------------------------------
@brief ~~> void find_score_offset_MS(int *seq_score, char* seq_temp)
The Function will find the best score, offset, k.
the answer will be applied to seq_score array.
seq_score[0] = Best Score value.
seq_score[1] = Best Offset value.
seq_score[2] = Best k value.

@param param1 -- int *seq_score: an array of 3 elements.
@param param1 -- char* seq_temp : the current seq to work on
@return: void
-----------------------------------------------------------------------*/
void find_score_offset_MS(int index, char* temp)
{
int temp_score = 0;

for (int k = worker_seq_arr[index].seq_len ; k >= 0 ; k--)
{
MS(temp, k);

for (int offset = 0; offset <= init_pack.seq1_len - worker_seq_arr[index].seq_len; offset++)
{
temp_score = check_score(temp, offset);
#ifdef DEBUG
printf("~~> score=%d \n", temp_score);
#endif
if (temp_score > worker_seq_arr[index].score)
{
worker_seq_arr[index].score = temp_score; // alignment score
worker_seq_arr[index].offset = offset;    // offset
worker_seq_arr[index].k = k;              // K
}
}
}
}
/*-----------------------------------------------------------------------
@brief ~~> void MS(char* seq, int k)
The Function will change the current seq with a given K.

@param param1 -- char* seq : the current seq to work on.
@param param1 -- int k : the current k.
@return: void
-----------------------------------------------------------------------*/
void MS(char *seq, int k)
{
if (k == strlen(seq))
{
return;
}
else
{
seq[k] = (seq[k] - 'A' + 1) % 26 + 'A';
}
}

/*-----------------------------------------------------------------------
@brief ~~> int check_score(char* seq, int offset)
The Function will calculate the score for a specific K and Offset.

@param param1 -- char* seq : the current seq to work on.
@param param1 -- int offset : the current offset.
@return: int value : the best score for the current seq.
-----------------------------------------------------------------------*/
int check_score(char *seq, int offset)
{
int value = 0;
int seq_len = strlen(seq);
for (int i = 0; i < seq_len; i++)
{    
value += init_pack.score_table[(init_pack.seq1[offset + i] - 'A')][(seq[i] - 'A')];
#ifdef DEBUG
printf("~~> table slot: %d * %d + %d\n",(seq1[offset+i] - 'A'),NUMBER_OF_LETTERS, (seq[i] - 'A'));
printf("~~> value so far = %d\n",value);
#endif
}
return value;
}

/*-----------------------------------------------------------------------
@brief  ~~> void exit_safely()
Function to free all the memory allocated in the program.

@return: void
-----------------------------------------------------------------------*/

void exit_safely(int myRank)
{
free(seq1);
if(myRank == MASTER)
{
free(master_seq_arr);
}else{
free(worker_seq_arr);
}
}

/*-----------------------------------------------------------------------
@brief  ~~> void toUpperCase(char *str)
Function to make a string letters all upper case.

@param param1 -- char* str ~> the string to be upper cased
@return: void
-----------------------------------------------------------------------*/
void toUpperCase(char *str)
{
if (str == NULL)
return;

for (int i = 0; str[i]; i++)
{
str[i] = toupper((unsigned char)str[i]);
}
}

/*-----------------------------------------------------------------------
@brief  ~~> int read_input_seq()
This function reads the input from "stdin" and initializing
the variables needed.
The input template:
1.  Seq1 (String): (The main series).
its the series that all the other serieses will be compared to.
Assumptions : Seq1 maximum lenth is 3000.
2.  number_of_sequences (int) : this will tell how many Strings
there is to read (Seq(2) , Seq(3) ... Seq(n)).
3.  Seq(1..n) : all the serieses line by line.
(i.e) :
ABBDAB
3
ADC
BDE
AAB

@return : int (Success\Fail)
-----------------------------------------------------------------------*/
int read_input_seq()
{
seq1 = handle_string_from_input();
init_pack.seq1_len = strlen(seq1);
strcpy(init_pack.seq1, seq1);
init_pack.seq1[init_pack.seq1_len] = '\0';

// Read the num_of_seqs value
if (fscanf(stdin, "%d", &init_pack.num_of_seqs) != 1)
{
printf("Failed to read input_number.\n");
return 1;
}
#ifdef DEBUG
printf("~~> Seq1 = %s , len = %d\n", init_pack.seq1, init_pack.seq1_len);
printf("~~> num_of_seq = %d \n", init_pack.num_of_seqs);
#endif
fgetc(stdin); // (read the "\n" that come after the number)
master_seq_arr = (Seq_Info *)malloc(init_pack.num_of_seqs * sizeof(Seq_Info));
if (master_seq_arr == NULL)
{
fprintf(stderr, "Memory allocation failed.\n");
exit(1);
}
// Read the seq_array
char *temp;
for (int i = 0; i < init_pack.num_of_seqs; i++)
{
temp = handle_string_from_input();
master_seq_arr[i].seq_len = strlen(seq1);
strcpy(master_seq_arr[i].seq, temp);
master_seq_arr[i].seq[init_pack.seq1_len] = '\0';
master_seq_arr[i].seq_len = strlen(master_seq_arr[i].seq);
master_seq_arr[i].offset = 0;
master_seq_arr[i].k = 0;
master_seq_arr[i].score = INITSCORE;
}
free(temp);
#ifdef DEBUG
for (int i = 0; i < init_pack.num_of_seqs; i++) // Print all the seqs
printf("~~> %s \n", seq_arr[i].seq);
#endif
return 0;
}

/*-----------------------------------------------------------------------
@brief  ~~> char* handle_string_from_input()
Function to Handle the string input:
this function allocates memory for a 3000 long string and
after that cut's down the unused memory.

@return: char*
-----------------------------------------------------------------------*/
char *handle_string_from_input()
{
char *temp = (char *)malloc(SEQ_1_MAX_LEN * sizeof(char));
if (temp == NULL)
{
fprintf(stderr, "Memory allocation failed.\n");
exit(1);
}
// Read SEQ
if (fgets(temp, SEQ_1_MAX_LEN, stdin) == NULL)
{
printf("Failed to read input_string.\n");
exit(1);
}
int temp_len = strlen(temp);
if (temp_len > 0 && temp[temp_len - 1] == '\n')
{
temp[temp_len - 1] = '\0';
}
char *str = strdup(temp);
toUpperCase(str);

return str;
}

/*-----------------------------------------------------------------------
@brief  ~~>void read_score_table(int argc, char **argv)
Function to initialize the score table.
the function checks if there is an argument passed to the program
and read the values from there to the score table.
-if there is no argument passed to it, the function will initialize
the score table with "0" exept the diagonal line (will be "1").
-if there is not enogh values in the txt file then the function
will padd the score table with "0".

@param param1 -- int argc ~> how many arguments passed at execution
@param param2 -- char** argv ~> an array of strings of the arguments
@ return: void
-----------------------------------------------------------------------*/
void read_score_table(int argc, char **argv)
{
if (argc == 2)
{
// Read Score table from a text file.
char *fileName = argv[1];
load_score_table_from_text_file(fileName);
}
else
{
// Default option (diagonal = 1, the rest = 0).
for (int i = 0; i < NUMBER_OF_LETTERS; i++)
init_pack.score_table[i][i] = 1;
}
#ifdef DEBUG
print_score_table();
#endif
}

int load_score_table_from_text_file(const char *fileName)
{
FILE *fp;
int count_v = 0;
int value;
fp = fopen(fileName, "r");
if (fp == NULL)
{
perror("Error opening file");
exit(1);
}
for (int i = 0; i < NUMBER_OF_LETTERS; i++)
{
for (int j = 0; j < NUMBER_OF_LETTERS; j++)
{
if (fscanf(fp, "%d", &value) == 1)
{
count_v++;
init_pack.score_table[i][j] = value;
}
else
{
printf("no more values\n");
break;
}
}
}
#ifdef DEBUG
if (count_v != NUMBER_OF_LETTERS * NUMBER_OF_LETTERS)
printf("%d values appear in the text file (expected %d values, (padding with \"0\"))\n",
count_v, NUMBER_OF_LETTERS * NUMBER_OF_LETTERS);
#endif
fclose(fp);
return 0;
}

/*-----------------------------------------------------------------------
@brief  ~~> void print_score_table()
Function to Print the score table.

@ return: void
-----------------------------------------------------------------------*/
void print_score_table()
{
for (int i = 0; i < NUMBER_OF_LETTERS; i++)
{
for (int j = 0; j < NUMBER_OF_LETTERS; j++)
{
printf("%4d", init_pack.score_table[i][j]);
}
putchar('\n');
}
}

void skip_white_space()
{
int c;
while (1)
{
if ((c = getchar()) == '\n')
lineno++;
else if (isspace(c))
continue;
else if (c == EOF)
break;
else
{
ungetc(c, stdin);
break;
}
}
}