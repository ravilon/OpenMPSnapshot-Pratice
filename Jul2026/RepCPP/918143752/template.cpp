#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <fstream>
#include <omp.h>
#include <mpi.h>
#include "template.cuh"

void read_matrix_file(mat_struct &matrix, int block_size, string mat_file)
{
ifstream file(mat_file);
matrix.exist = 1;
file >> matrix.height >> matrix.width >> matrix.num_blocks;
int t1, t2;

matrix.vec.resize(matrix.num_blocks * block_size * block_size, 0);

for (int i = 0; i < matrix.num_blocks; i++)
{
int loc = i * block_size * block_size;

file >> t1 >> t2;

for (int j = 0; j < block_size * block_size; j++)
{
file >> matrix.vec[loc + j];
}
matrix.mat_map[{t1, t2}] = loc;
}

file.close();
}

vector<u_int64_t> serialize_to_vec(mat_struct &matrix)
{
vector<u_int64_t> v;
v.push_back(matrix.exist);
v.push_back(matrix.height);
v.push_back(matrix.width);
v.push_back(matrix.num_blocks);

if (matrix.exist == 0)
{
return v;
}

for (auto &[p, loc] : matrix.mat_map)
{
v.push_back(p.first);
v.push_back(p.second);
v.push_back(loc);
}
v.insert(v.end(), make_move_iterator(matrix.vec.begin()), make_move_iterator(matrix.vec.end()));
return v;
}

mat_struct deserialize_to_map(vector<uint64_t> &vec, int block_size)
{
mat_struct matrix;

matrix.exist = vec[0];
if (vec[0] == 0)
{
matrix.num_blocks = 0;
return matrix;
}

matrix.height = vec[1];
matrix.width = vec[2];
matrix.num_blocks = vec[3];

for (int i = 0; i < matrix.num_blocks; i++)
{
int r = vec[4 + 3 * i];
int c = vec[4 + 3 * i + 1];
int l = vec[4 + 3 * i + 2];

matrix.mat_map[{r, c}] = l;
}

matrix.vec.insert(matrix.vec.end(), make_move_iterator(vec.begin() + 4 + 3 * matrix.num_blocks), make_move_iterator(vec.end()));

return matrix;
}

mat_struct multipy_matrices_part(int rank, int size, int num_matrices, int block_size, string foldername)
{
int base = num_matrices / size;
int remainder = num_matrices % size;

int mat_start = rank * base + min(rank, remainder) + 1;
int mat_end = mat_start + base + (rank < remainder ? 1 : 0) - 1;

if (mat_start > mat_end)
{
mat_struct temp;
temp.exist = 0;
temp.height = 0;
temp.width = 0;
temp.num_blocks = 0;
return temp;
}

vector<mat_struct> mat_array(mat_end - mat_start + 1);

omp_set_num_threads(omp_get_num_procs());

#pragma omp parallel for
for (int i = mat_start; i <= mat_end; i++)
{
read_matrix_file(mat_array[i - mat_start], block_size, foldername + "/matrix" + to_string(i));
}

for (int i = 1; i <= mat_end - mat_start; i++)
{
mat_array[i] = multiply_matrices(mat_array[i - 1], mat_array[i], block_size);
}

return mat_array[mat_array.size() - 1];
}

int main(int argc, char **argv)
{
MPI_Init(&argc, &argv);

int rank, size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);

string test_folder = argv[1];
ifstream size_file(test_folder + "/size");
int num_matrices, block_size;
size_file >> num_matrices >> block_size;

mat_struct matrix_part = multipy_matrices_part(rank, size, num_matrices, block_size, test_folder);

vector<uint64_t> serialised_part_res = serialize_to_vec(matrix_part);
int local_count = serialised_part_res.size();

vector<int> recv_counts(size);
MPI_Gather(&local_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

vector<int> displs(size);
int total_count = 0;
if (rank == 0)
{
for (int i = 0; i < size; i++)
{
displs[i] = total_count;
total_count += recv_counts[i];
}
}

vector<uint64_t> recv_data;
if (rank == 0)
{
recv_data.resize(total_count);
}

MPI_Gatherv(serialised_part_res.data(), local_count, MPI_UINT64_T, recv_data.data(), recv_counts.data(), displs.data(), MPI_UINT64_T, 0, MPI_COMM_WORLD);

mat_struct prev_mat;
mat_struct new_mat;

if (rank == 0)
{
for (int i = 0; i < size; i++)
{
if (i == 0)
{
vector<uint64_t> chunk(recv_data.begin() + displs[i], recv_data.begin() + displs[i] + recv_counts[i]);
prev_mat = deserialize_to_map(chunk, block_size);
}
else
{
vector<uint64_t> chunk(recv_data.begin() + displs[i], recv_data.begin() + displs[i] + recv_counts[i]);
new_mat = deserialize_to_map(chunk, block_size);

if (new_mat.exist == 0)
{
continue;
}

prev_mat = multiply_matrices(prev_mat, new_mat, block_size);
}
}
}

vector<pair<int, int>> keys;
for (const auto &entry : prev_mat.mat_map)
{
bool non_zero = false;
int loc = prev_mat.mat_map[entry.first];

for (int i = 0; i < block_size * block_size; i++)
{
if (prev_mat.vec[loc + i] != 0)
{
non_zero = true;
break;
}
}

if (non_zero)
{
keys.push_back(entry.first);
}
}

sort(keys.begin(), keys.end());

ofstream outfile("matrix");
outfile << prev_mat.height << " " << prev_mat.width << endl;
outfile << keys.size() << endl;

for (const auto &key : keys)
{
outfile << key.first << " " << key.second << '\n';
int loc = prev_mat.mat_map[key];

for (int i = 0; i < block_size * block_size; i++)
{
outfile << prev_mat.vec[loc + i] << " ";
if ((i + 1) % block_size == 0)
{
outfile << '\n';
}
}
}

outfile.close();

MPI_Finalize();

return 0;
}
