#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <crypt.h>
#include <unistd.h>
#include <math.h>

char *decimalToAlphabet(int deci)
{
char *s = malloc(30);
char *r = malloc(30);
int i = 0;
while (deci-- != 0)
{
int alpha = deci % 26;
char ch = alpha + 97;
deci = deci / 26;
s[i++] = ch;
}
s[i] = '\0';
// Now reversing the string
int count = 0;
while (s[count] != '\0')
count++;
int lengthS = count - 1;
for (i = 0; i < count; ++i) {
r[i] = s[lengthS];
--lengthS;
}
r[i] = '\0';
return r;
}

char *fileReading(char username[], int nameLen)
{
//Reading file and extracting required line
FILE *fp;
char *line = NULL;
size_t len = 0;
ssize_t read;

// Reading /etc/shadow
fp = fopen("/etc/shadow", "r");
if (fp == NULL)
{
printf("File not found\n");
fclose(fp);
MPI_Finalize();
return NULL;
}

while ((read = getline(&line, &len, fp)) != -1)
{
//comparing username
int result = strncmp(username, line, nameLen);
if (result == 0)
{
break;
}
}
fclose(fp);

// target hash and its length
char *targetHash = malloc(98 * sizeof(char));
memcpy(targetHash, line + nameLen + 1, 98 * sizeof(char));

return targetHash;
}

int main(int argc, char *argv[])
{

int nprocs, rank;

MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

/*	Regarding MPI Send & Receive:				*/
/*	===========================================	*/
/*	array size communication on tag 		= 0 */
/*	array itself communication on tag 		= 1 */
/*	number to search communication on tag 	= 2 */
/*	abort message communication on tag 		= 3 */

// master
if (rank == 0)
{
// clearing the console
int sysRet = system("clear");
if (sysRet == -1)
{
printf("System Method Failed.\n");
return 0;
}

printf("PDC Project - Distributed Password Cracker\n");
printf("============================================\n\n");

// =============================================================================================
// STEP-1: Take username input
// =============================================================================================

// initializing variables
char username[30] = "waqar";
printf("Please enter Username: ");
if (scanf("%s", username))
{
}
printf("\nYour have entered %s.", username);
int nameLen = (int)(strlen(username));
printf(" It's length is %d.\n\n", nameLen);

// =============================================================================================
// STEP-2: Read /etc/shadow file
// =============================================================================================

char *targetHash = fileReading(username, nameLen);
int targetHashLen = (int)strlen(targetHash);

// =============================================================================================
// STEP-3: Distribute workloads
// =============================================================================================

int numDigits = 8, totalPermutations = pow(26, numDigits);
int iproc, chunkStart = 0, chunkLength = totalPermutations / (nprocs - 1), abort = 0;
MPI_Status status;

printf("Master: The hash to crack is: %s\n\n", targetHash);

// each process gets equal range i.e., (total permutations) / (#slaves)
for (iproc = 1; iproc < nprocs; iproc++)
{
if (iproc == nprocs - 1 && chunkStart + chunkLength < totalPermutations - 1)
{
chunkLength = chunkLength + (totalPermutations) - (chunkStart + chunkLength);
}
MPI_Send(&targetHashLen, 1, MPI_INT, iproc, 0, MPI_COMM_WORLD);			 // targetHashLen sent with tag 0
MPI_Send(targetHash, targetHashLen, MPI_CHAR, iproc, 1, MPI_COMM_WORLD); // targetHash sent with tag 1
MPI_Send(&chunkStart, 1, MPI_INT, iproc, 2, MPI_COMM_WORLD);			 // chunkStart sent with tag 2
MPI_Send(&chunkLength, 1, MPI_INT, iproc, 3, MPI_COMM_WORLD);			 // chunkLength sent with tag 3
chunkStart += chunkLength;
}

// =============================================================================================
// STEP-4: Sit tight and wait for good news
// =============================================================================================

MPI_Recv(&abort, 1, MPI_INT, MPI_ANY_SOURCE, 4, MPI_COMM_WORLD, &status); // abort message received with tag 4
printf("Master: Process %d has cracked the password!\n", status.MPI_SOURCE);
printf("Master: Informing all processes to abort!\n");
// sending abort messages
for (iproc = 1; iproc < nprocs; iproc++)
MPI_Send(&abort, 1, MPI_INT, iproc, 4, MPI_COMM_WORLD); // abort messages sent with tag 4
printf("============================================\n");
printf("Master: The password is...drum roll...\n\t%s\n", decimalToAlphabet(abort));
printf("============================================\n");
}

// slaves
else
{
// initializing variables
int chunkStart, chunkLength, abort = 0, targetHashLen;

// receiving initial instructions from master
MPI_Recv(&targetHashLen, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);			// receive targetHashLen with tag 0
char *targetHash = malloc(targetHashLen * sizeof(char));								// allocate memory according to targetHashLen
MPI_Recv(targetHash, targetHashLen, MPI_CHAR, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // receiving targetHash itself with tag 1
MPI_Recv(&chunkStart, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);				// receive chunkStart with tag 2
MPI_Recv(&chunkLength, 1, MPI_INT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);			// receive chunkLength with tag 3

printf("Process %d:\n \
\tHash to crack: %s\n \
\tChunkStart: %d\n \
\tChunkLength: %d\n\n",
rank, targetHash, chunkStart, chunkLength);

// using openmp to create 2 threads - one for processing and one for listening for abort message
#pragma omp parallel num_threads(2)
{
// thread-1: processing
if (omp_get_thread_num() == 0)
{
char *salt = malloc(12 * sizeof(char));
memcpy(salt, targetHash, 12 * sizeof(char));
int i = chunkStart;
for (; i < chunkStart + chunkLength; i++)
{
// abandon rest of searching if abort message received
if (abort != 0)
{
printf("\nProcess %d: Aborting!", rank);
break;
}
printf("Process %d: Attempt No.(%d).\n", rank, i);
// if found: notify master
if (strcmp(crypt(decimalToAlphabet(i), salt), targetHash) == 0)
{
printf("Process %d: I have cracked the password :-)\n", rank);
MPI_Send(&i, 1, MPI_INT, 0, 4, MPI_COMM_WORLD); // abort message with tag 3
break;
}
}
}
// thread-2: listening for abort message
else if (omp_get_thread_num() == 1)
{
MPI_Recv(&abort, 1, MPI_INT, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // waiting for possible abort message from master with tag 4
}
}
}

MPI_Finalize();
return 0;
}
