#include <random>
#include "modify.cuh"
#include <chrono>
#include <fstream>
#include <algorithm>
#include <cassert>
#include <omp.h>
#include <iostream>
#include <unordered_map>

void print(vector<vector<int>> &matrix)
{
for (int i = 0; i < matrix.size(); i++)
{
for (int j = 0; j < matrix[0].size(); j++)
cout << matrix[i][j] << ' ';
cout << endl;
}
cout << endl;
}

bool check(vector<vector<vector<int>>> &upd_matrices, vector<vector<vector<int>>> &org_matrices)
{
cout << "checking" << endl;
if (org_matrices.size() ^ upd_matrices.size())
{
cout << "Test case failed 1" << endl;
return false;
}

for (int i = 0; i < org_matrices.size(); i++)
{
const auto &matrix = upd_matrices[i];
const auto &original = org_matrices[i];
int rows = matrix.size(), cols = matrix[0].size();
for (int i = 0; i < rows; i++)
{
for (int j = 0; j < cols; j++)
{
if (i > 0 && matrix[i][j] < matrix[i - 1][j])
{
cout << "Test case failed 2" << endl;
return false;
}
if (j > 0 && matrix[i][j] < matrix[i][j - 1])
{
cout << "Test case failed 3" << endl;
return false;
}
}
}
vector<int> freqv(1e9);
unordered_map<int, int> freq;
for (int i = 0; i < rows; i++)
{
for (int j = 0; j < cols; j++)
{
freqv[original[i][j]]++; // Increment for original matrix
freqv[matrix[i][j]]--;   // Decrement for updated matrix
}
}

for (const auto &it : freqv)
{
if (it != 0)
{
cout << "Test case failed 4" << endl;
return false;
}
}
cout << "Test case passed" << endl;
}
cout << "All matrix passed" << endl;
return true;
}

vector<vector<int>> gen_matrix(int range, int rows, int cols)
{
assert(range <= (int)1e9);
vector<vector<int>> matrix(rows, vector<int>(cols));
omp_set_num_threads(16);
#pragma omp parallel
{
std::mt19937 gen(omp_get_thread_num());
std::uniform_int_distribution<int> dist(0, range);
#pragma omp for
for (int i = 0; i < rows; i++)
{
long long foo = dist(gen);
for (int j = 0; j < cols; j++)
{
matrix[i][j] = 1 + ((foo ^ j) % range);
}
}
}
return matrix;
}

int main()
{
ofstream file("results.txt");

int range = 100000000;
int rows = 10000;
int cols = 100000;
int num_matrices = 1;

vector<vector<vector<int>>> matrices;
vector<int> ranges;
for (int i = 0; i < num_matrices; i++)
{
matrices.push_back(gen_matrix(range, rows, cols));
file << "mat " << i << " generated" << endl;
ranges.push_back(range);
}

cout << "Matrices generated" << endl;
auto start = std::chrono::high_resolution_clock::now();

vector<vector<vector<int>>> upd_matrices = modify(matrices, ranges);

auto end = std::chrono::high_resolution_clock::now();
chrono::duration<double, std::milli> duration = end - start;

cout << "Matrices modified" << endl;
file << duration.count() << " ms" << endl;
cout << duration.count() << " ms" << endl;

bool corr = check(upd_matrices, matrices);
if (corr)
file << "Test Passed\n";
else
file << "Test Failed\n";
return 0;
}
