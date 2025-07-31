#include <omp.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#include <numeric> // accumulate
#include <algorithm> // min & max
#include <random> // mt19937

using namespace std;

vector<vector<int> > matrixA;
vector<vector<int> > matrixB;

vector<vector<int> > multiplyMatrices(const vector<vector<int> > &matrixA, const vector<vector<int> > &matrixB) {
    const int rowsA = matrixA.size();
    const int colsA = matrixA[0].size();
    const int rowsB = matrixB.size();
    const int colsB = matrixB[0].size();
    if (colsA != rowsB) {
        throw runtime_error("Error: Matrices are incompatible for multiplication. "
            "Columns of the first matrix must equal rows of the second.");
    }

    vector<vector<int> > resultMatrix(rowsA, vector<int>(colsB, 0));

    // Parallelize the outer two loops using OpenMP's collapse clause.
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            // Use a private variable for accumulating the sum for each element.
            int sum = 0;
            for (int k = 0; k < colsA; k++) {
                sum += matrixA[i][k] * matrixB[k][j];
            }
            // Assign the accumulated sum to the result matrix element.
            resultMatrix[i][j] = sum;
        }
    }

    return resultMatrix;
}

vector<vector<int> > generateMatrix(int rows, int cols, int seed) {
    // A random number generator engine with the given seed
    mt19937 generator(seed);
    // A uniform distribution between a suitable range (e.g., 1 to 100)
    uniform_int_distribution<int> distribution(1, 100);
    vector<vector<int> > matrix(rows, vector<int>(cols));
    // Fill the matrix with random numbers
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = distribution(generator);
        }
    }
    return matrix;
}

void program(const string &output_file) {
    // Call multiplyMatrices() within a try-catch block to handle potential errors
    try {
        const vector<vector<int> > result = multiplyMatrices(matrixA, matrixB);

        if (const size_t elements = result.size() * result[0].size(); elements < 100) {
            // Print the result if no exception was thrown
            cout << "Resultant Matrix:" << endl;
            for (const auto &row: result) {
                for (const int val: row) {
                    cout << val << " ";
                }
                cout << endl;
            }
        } else {
            cout << "Resultant Matrix Size: " << elements << endl;
            // Save the result to a file
            if (ofstream outfile(output_file); outfile.is_open()) {
                for (const auto &row: result) {
                    for (const int val: row) {
                        outfile << val << " ";
                    }
                    outfile << endl;
                }
                outfile.close();
            } else {
                cerr << "Error: Could not open output file " << output_file << endl;
            }
        }
    } catch (const runtime_error &error) {
        cerr << error.what() << endl;
    }
}

int measure(const int num_threads, const string &output_file) {
    omp_set_num_threads(num_threads);

    constexpr int warmups = 3;
    constexpr int runs = 8;

    vector<double> exec_times;

    // Warm-up
    for (int i = 0; i < warmups; i++) {
        cout << "Warm-up Round " << i + 1 << "/" << warmups << " with " << num_threads << " threads\n";
        program("discard_warmup.txt");
    }

    // Measurement
    for (int i = 0; i < runs; i++) {
        cout << "Round " << i + 1 << "/" << runs << " with " << num_threads << " threads\n";
        const double start_time = omp_get_wtime();
        program("result_matrix.txt");
        const double end_time = omp_get_wtime();
        exec_times.push_back(end_time - start_time);
    }

    // Calculate statistics
    const double sum = accumulate(exec_times.begin(), exec_times.end(), 0.0);
    const double average = sum / exec_times.size();
    const double min_time = *ranges::min_element(exec_times);
    const double max_time = *ranges::max_element(exec_times);
    const double sq_sum = inner_product(exec_times.begin(), exec_times.end(), exec_times.begin(), 0.0);
    const double stdev = sqrt(sq_sum / exec_times.size() - average * average);

    // Write output to file
    if (ofstream outfile(output_file, ios::app); outfile.is_open()) {
        outfile << num_threads << " "
                << fixed << setprecision(6) << average << " "
                << fixed << setprecision(6) << min_time << " "
                << fixed << setprecision(6) << max_time << " "
                << fixed << setprecision(6) << stdev << "\n";
        outfile.close();
    } else {
        cerr << "Error: Could not open output file " << output_file << endl;
        return 1;
    }
    return 0;
}

int main(const int argc, char *argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <num_threads> <output_file>\n";
        return 1;
    }

    matrixA = generateMatrix(2048, 64, 20482048);
    matrixB = generateMatrix(64, 4096, 40964096);

    try {
        const int num_threads = stoi(argv[1]);
        const string output_file = argv[2];
        return measure(num_threads, output_file);
    } catch (const exception &e) {
        cerr << "Error: Invalid input arguments: " << e.what() << endl;
        return 1;
    }
}
