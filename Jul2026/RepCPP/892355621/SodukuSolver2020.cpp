#include <iostream>
#include <ctime>
#include <omp.h>
#include <chrono>
using namespace std;

#define N 20
int zero_counter=0;
int nonzero_counter = 0;

void print(int arr[N][N])
{

for (int i = 0; i < N; i++)
{
for (int j = 0; j < N; j++)
cout << arr[i][j] << " ";
cout << endl;
}

}

bool isSafe(int grid[N][N], int row, int col, int num)
{

for (int x = 0; x < N; x++)
if (grid[row][x] == num)
return false;

for (int x = 0; x < N; x++)
if (grid[x][col] == num)
return false;

int startRow = row - row % 4, startCol = col - col % 5;

for (int i = 0; i < 4; i++)
for (int j = 0; j < 5; j++)
if (grid[i + startRow][j + startCol] == num)
return false;


return true;
}

bool solveSudoku(int grid[N][N], int row, int col)
{


if (row == N - 1 && col == N)
return true;

if (col == N) {
row++;
col = 0;
}

if (grid[row][col] > 0)
return solveSudoku(grid, row, col + 1);


for (int num = 1; num <= N; num++)
{
if (isSafe(grid, row, col, num))
{
grid[row][col] = num;

if (solveSudoku(grid, row, col + 1))
return true;
}

grid[row][col] = 0;
}


return false;
}

int main()
{
int grid[N][N] = { {1,0,0,0,0,0,10,0,6,9,0,17,11,0,0,20,3,7,0,18},
{0,10,0,0,0,5,2,0,11,0,13,0,7,0,18,8,0,15,0,0},
{ 0,13,0,0,0,1,4,12,15,8,10,19,0,9,0,0,0,11,0,5},
{0,0,0,0,14,18,0,0,7,0,4,0,15,8,0,9,19,0,10,0},
{0,0,0,20,18,4,0,0,8,1,0,0,0,16,0,0,11,14,17,2},
{4,12,0,0,1,10,19,6,0,0,0,0,14,0,2,18,7,20,0,13},
{10,19,0,9,16,0,0,0,14,5,3,7,20,0,13,0,15,0,0,4},
{0,17,0,0,5,0,3,7,0,18,0,15,8,0,4,16,0,9,0,0},
{19,0,9,0,10,0,0,14,0,0,7,20,18,0,3,4,8,0,15,0},
{12,0,0,0,4,19,0,0,16,10,11,0,0,2,0,13,20,18,7,0},
{17,0,14,5,2,0,7,0,18,0,15,0,0,0,0,10,0,0,0,0},
{3,0,20,0,0,0,0,0,1,4,6,9,16,10,19,2,0,0,0,17},
{0,0,18,0,3,15,0,0,4,0,0,16,10,19,0,17,5,2,0,11},
{0,8,0,0,0,6,9,16,10,19,0,5,0,17,0,0,18,0,0,7},
{6,9,0,0,19,11,14,0,0,17,20,0,0,0,7,0,1,0,0,15},
{0,0,0,0,0,0,20,0,13,3,0,0,4,0,15,0,0,0,9,0},
{0,18,13,0,0,8,0,4,0,0,0,0,19,6,0,11,2,0,5,0},
{0,0,4,0,15,0,0,10,0,6,0,0,17,11,0,7,13,3,0,0},
{9,0,10,0,6,0,5,2,17,0,0,13,0,0,20,0,4,0,0,8},
{0,5,2,0,11,0,18,13,0,0,1,0,0,0,8,6,0,19,0,9} };

auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(2)
for (int x = 0; x < N; x++) {
for (int y = 0; y < N; y++) {
if (grid[x][y] == 0) {
#pragma omp atomic
zero_counter++;  // Count zero elements
} else {   
#pragma omp atomic
nonzero_counter++;  // Count non-zero elements
}
}
}
cout << "Number of zeroes in grid: (Parallel:) " << zero_counter << endl;
cout << "Number of non-zeros in grid: (Parallel:) " << nonzero_counter << endl;


if (solveSudoku(grid, 0, 0))
print(grid);
else
cout << "No solution exists" << endl;
auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

cout << "Total execution time for the whole program:" << duration.count() << " seconds" << endl;
}
