//---------------------------//
//   KNN  - main func       //
//--------------------------//

/*
1. Dataset Building
- need:
- word frequency for each book
- label for class
- book id to identify book in dataset


- find total unique words in the full dataset among all books
- union of all the books unique word sets

2. Distance Calculation
- cosine function (parallel)

3. sort KNN into array
- use distance from cosine function.
- keep track of label and/or book_id associated with this distance

4. Majority Function
- find majority of KNN array
- assign label of  majority to queried book

5. Check Accuracy of predicted label(s) //TODO
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <typeinfo>
#include <vector>
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <string.h>

#define _NUM_THREADS 6

using namespace std;

void printMatrix(vector<vector<float>> countMatrix)
{
// To see if counts were properly transferred into countMatrix
for (int i = 0; i < countMatrix.size(); i++)
{
cout << countMatrix[i][0] << " " << countMatrix[i][1] << " " << countMatrix[i][2] << " " << countMatrix[i][3] << endl;
}
}

void readInMatrix(vector<vector<string>> &fullMatrix, int &rowCount, int &colCount)
{
ifstream myFile, rowsFile, colsFile;
rowsFile.open("data/rows.txt");
colsFile.open("data/cols.txt");
myFile.open("data/matrix.csv");

rowsFile >> rowCount;
rowsFile.close();
colsFile >> colCount;
colsFile.close();

while (myFile.good())
{
vector<string> row;
string line;
getline(myFile, line);
stringstream iss(line);

for (int cols = 0; cols < colCount + 1; cols++)
{
string val;
getline(iss, val, ',');
row.push_back(val);
}
fullMatrix.push_back(row);
}
myFile.close();
}

void loadWordCountMatrix(vector<vector<float>> &countMatrix, vector<vector<string>> fullMatrix, vector<string> &bookList, int rowCount, int colCount)
{
for (int j = 2; j < colCount + 1; j++)
{
bookList.push_back(fullMatrix[0][j]);
//cout << bookList[j - 2];
for (int i = 1; i < rowCount + 1; i++)
{
int val = stoi(fullMatrix[i][j]);
countMatrix[i - 1].push_back(val);
}
}
//cout << endl;
}

void cosineSimilarity(vector<vector<float>> countMatrix, vector<float> &cosList, vector<string> bookList, int rank, int querybook)
{
int i, j;
for (int j = 0; j < countMatrix[1].size(); j++)
{
float den1 = 0, den2 = 0, num = 0;
#pragma omp parallel for reduction(+  : num, den1, den2)

for (int i = 0; i < countMatrix.size(); i++)
{
num += (countMatrix[i][rank] * countMatrix[i][countMatrix[1].size() - 1]); //dot product
den1 += pow(countMatrix[i][rank], 2);                                      //magnitude of known book
den2 += pow(countMatrix[i][countMatrix[1].size() - 1], 2);                 //magnitude of query book
}
cosList.push_back(num / (sqrt(den1 * den2))); //cosine similarity between this book and query book
}
//print cosine similarity between all books and querybook
if (rank != querybook) //skip printing querybook similarity to itself
cout << bookList[rank] << " cos similarity to " << bookList[querybook]
<< ": " << cosList[rank] << endl;
}

// Author: Wes Kendall
// Copyright 2013 www.mpitutorial.com
// This code is provided freely with the tutorials on mpitutorial.com. Feel
// free to modify it for your own use. Any distribution of the code must
// either provide a link to www.mpitutorial.com or keep this header intact.

typedef struct
{
int comm_rank;
int label;
float number; // distance
} CommRankNumber;

// Gathers numbers for TMPI_Rank to process zero. Allocates enough space given the MPI datatype and
// returns a void * buffer to process 0. It returns NULL to all other processes.
void *gather_numbers_to_root(void *number, MPI_Datatype datatype, MPI_Comm comm)
{
int comm_rank, comm_size;
MPI_Comm_rank(comm, &comm_rank);
MPI_Comm_size(comm, &comm_size);

// Allocate an array on the root process of a size depending on the MPI datatype being used.
int datatype_size;
MPI_Type_size(datatype, &datatype_size);
void *gathered_numbers;
if (comm_rank == 0)
{
gathered_numbers = new int[datatype_size * comm_size];
}

// Gather all of the numbers on the root process
MPI_Gather(number, 1, datatype, gathered_numbers, 1, datatype, 0, comm);

return gathered_numbers;
}

// A comparison function for sorting float CommRankNumber values
int compare_float_comm_rank_number(const void *a, const void *b)
{
CommRankNumber *comm_rank_number_a = (CommRankNumber *)a;
CommRankNumber *comm_rank_number_b = (CommRankNumber *)b;
if (comm_rank_number_a->number > comm_rank_number_b->number)
{
return -1;
}
else if (comm_rank_number_a->number < comm_rank_number_b->number)
{
return 1;
}
else
{
return 0;
}
}

// This function sorts the gathered numbers on the root process and returns an array of
// ordered by the process's rank in its communicator. Note - this function is only
// executed on the root process.
CommRankNumber *get_sorted(void *gathered_numbers, int gathered_number_count, MPI_Datatype datatype)
{
int datatype_size;
MPI_Type_size(datatype, &datatype_size);

// Convert the gathered number array to an array of CommRankNumbers. This allows us to
// sort the numbers and also keep the information of the processes that own the numbers
// intact.
CommRankNumber *comm_rank_numbers = new CommRankNumber[gathered_number_count * sizeof(CommRankNumber)];
int i;
for (i = 0; i < gathered_number_count; i++)
{
comm_rank_numbers[i].comm_rank = i;
memcpy(&(comm_rank_numbers[i].number), gathered_numbers + (i * datatype_size), datatype_size);
}

// Sort the comm rank numbers
qsort(comm_rank_numbers, gathered_number_count, sizeof(CommRankNumber), &compare_float_comm_rank_number);

// Now that the comm_rank_numbers are sorted, create an array of rank values for each process. The ith
// element of this array contains the rank value for the number sent by process i.
int *nearest_neighbor = (int *)malloc(sizeof(int) * gathered_number_count);

//return sorted array
return comm_rank_numbers;
}

// Gets the rank of the recv_data, which is of type datatype. The rank is returned
// in send_data and is of type datatype.
int TMPI_Rank(void *send_data, void *recv_data, MPI_Datatype datatype, MPI_Comm comm, int k, vector<string> bookList, int querybook, int *class_labels, int num_classes)
{
// Check base cases first - Only support MPI_FLOAT
if (datatype != MPI_FLOAT)
{
return MPI_ERR_TYPE;
}

int comm_size, comm_rank;
MPI_Comm_size(comm, &comm_size);
MPI_Comm_rank(comm, &comm_rank);

float next, current; //distance of neighbor
string name;         //book name from bookList
int next_loc;        //rank (MPI) of book
int label;           // class label of book

// To calculate the rank, we must gather the numbers to one process, sort the numbers, and then
// scatter the resulting rank values. Start by gathering the numbers on process 0 of comm.
void *gathered_numbers = gather_numbers_to_root(send_data, datatype, comm);

// Get the ranks of each process
CommRankNumber *nearest_neighbor = NULL;
if (comm_rank == 0)
{
nearest_neighbor = get_sorted(gathered_numbers, comm_size, datatype);
current = nearest_neighbor[0].number; // unknown book (query book)

int max_count = 0; //majority label count
int max;           // the nearest neighbors are at the end of the sorted list

cout << "\nShowing only the nearest k = " << k << " nearest neighbors...\n\n";

int class_count[num_classes];
memset(class_count, 0, num_classes * sizeof(int));

// print KNN by considering only sorted values up to k
for (int i = 1; i < k + 1; i++)
{
name = bookList[nearest_neighbor[i].comm_rank];
next_loc = nearest_neighbor[i].comm_rank;
next = nearest_neighbor[i].number;
label = class_labels[nearest_neighbor[i].comm_rank];
cout << "nearest neighbor " << i << "\n\t| Rank " << next_loc << " = " << next;
cout << "\n\t| Book (" << name << ") has label " << label << endl
<< endl; //TODO: pass book ID

class_count[label]++;
}
// majority function
for (int i = 0; i < num_classes; i++)
{
if (max_count < class_count[i])
{
max = i;
max_count = class_count[i];
}
}

cout << "RESULT: using k = " << k << ",\n\tPredicted label class is " << max << " -- if no majority, picked one" << endl;
}

// Do clean up
if (comm_rank == 0)
{
free(gathered_numbers);
free(nearest_neighbor);
}

return 0;
}

int main(int argc, char *argv[])
{

enum author
{
fitzgerald = 1,
melville = 2,
shakespeare = 3,
unknown = 0
}; // possible author labels

int class_labels[4] = {shakespeare, fitzgerald, melville, shakespeare}; //TODO: read in from label file

int k;
k = 3;                             //TODO: remove hardcoded k value
int querybook = 3;                 //query book index
class_labels[querybook] = unknown; //

int rank;
int size;

omp_set_num_threads(_NUM_THREADS); //!!: unknown if working

MPI_Init(&argc, &argv); //!!: change to mpi_init_thread for hybrid?
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

// needed in matrix loading and cossine calculation
vector<vector<string>> fullMatrix;
vector<vector<float>> countMatrix;
vector<string> bookList; // passed to knn sort for printing book name
vector<float> cosList;   // passed to knn sort (TMPI_rank)
int rowCount, colCount;

// Function to read the matrix.csv into a 2d vector
readInMatrix(fullMatrix, rowCount, colCount);
// Resizes count matrix so it can be populated
countMatrix.resize(rowCount);
// Function to take all the word counts from each book, convert them to integers, then load them into new 2d array
loadWordCountMatrix(countMatrix, fullMatrix, bookList, rowCount, colCount);
// Function to print the countMatrix to make sure it was loaded properly (only works for 4 books atm)
/* printMatrix(countMatrix); */
// Function to run cosine similarity on classified books vs unclassifed book
cosineSimilarity(countMatrix, cosList, bookList, rank, querybook);

// KNN sorting function.
//sorts all ranks then prints only the nearest k values
//majority found from k values, if any, to print as predicted label
TMPI_Rank(&cosList[rank], &rank, MPI_FLOAT, MPI_COMM_WORLD, k, bookList, querybook, class_labels, sizeof(author)); //TODO: pass class label

MPI_Finalize();

return 0;
}