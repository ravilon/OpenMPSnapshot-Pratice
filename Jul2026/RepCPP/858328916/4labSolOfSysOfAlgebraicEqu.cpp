#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <iomanip>

using namespace std;

//  
const int NUM_RUNS = 100; //  

//      -   OpenMP
void gaussJordan(vector<vector<double>>& A, vector<double>& b, int num_threads) {
    int n = A.size();

    for (int i = 0; i < n; i++) {
        //   i   
        double diag = A[i][i];

#pragma omp parallel for num_threads(num_threads)
        for (int j = i; j < n; j++) {
            A[i][j] /= diag;
        }
        b[i] /= diag;

        //     i   ,   i
#pragma omp parallel for num_threads(num_threads)
        for (int k = 0; k < n; k++) {
            if (k != i) {
                double factor = A[k][i];
                for (int j = i; j < n; j++) {
                    A[k][j] -= factor * A[i][j];
                }
                b[k] -= factor * b[i];
            }
        }
    }
}

//     
void displayProgress(int current, int total, double avg_time) {
    int barWidth = 50;  //  -
    double progress = (double)current / total;  //    

    cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) cout << "=";
        else if (i == pos) cout << ">";
        else cout << " ";
    }
    cout << "] " << int(progress * 100.0) << "% ";
    cout << "Completed: " << current << "/" << total << " ";
    cout << "Avg time: " << avg_time << " sec\r";
    cout.flush();  //  
}

//    
void printResults(const vector<vector<double>>& A, const vector<double>& b, const vector<double>& result) {
    int n = A.size();
    cout << fixed << setprecision(2); //   

    cout << "Initial matrix A:\n";
    for (const auto& row : A) {
        for (double val : row) {
            cout << setw(10) << val << "\t"; //    
        }
        cout << endl;
    }

    cout << "\nInitial vector b:\n";
    for (double val : b) {
        cout << setw(10) << val << "\t"; //    
    }
    cout << endl;

    cout << "\nResult vector x:\n";
    for (double val : result) {
        cout << setw(10) << val << "\t"; //    
    }
    cout << endl;

    cout << std::resetiosflags(std::ios::fixed);
}

int main() {
    srand(time(0));  //    
    int num_threads = 0;
    int N = -1;

    //          
    while (true) {
        cout << "Enter number of threads (0 to exit): ";
        cin >> num_threads;
        if (num_threads == 0) {
            cout << "Program terminated.\n";
            return 0;
        }

        //   

        while (true)
        {
            cout << "Enter system size (0 to re-enter number of threads): ";
            cin >> N;
            if (N == 0) {
                break;  //     
            }

            //    
            vector<vector<double>> A(N, vector<double>(N));
            vector<double> b(N);
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    A[i][j] = rand() % 100 + 1;  //    1  100
                }
                b[i] = rand() % 100 + 1;
            }

            //   
            char display_choice;
            cout << "Do you want to display initial data and results? (Y/N): ";
            cin >> display_choice;

            //    
            display_choice = toupper(display_choice);
            if (display_choice != 'Y' && display_choice != 'N') {
                cout << "Invalid input! Defaulting to 'N'." << endl;
                display_choice = 'N';
            }
            std::cout << "\033[A\033[2K";

            //     100 
            double total_time = 0.0;

            vector<double> final_result(N);

            for (int i = 0; i < NUM_RUNS; i++) {
                //    A   b   
                vector<vector<double>> A_copy = A;
                vector<double> b_copy = b;

                double start_time = omp_get_wtime();  //  
                gaussJordan(A_copy, b_copy, num_threads);
                double end_time = omp_get_wtime();    //  

                total_time += (end_time - start_time);  //  

                //  
                double avg_time = total_time / (i + 1);  //      
                displayProgress(i + 1, NUM_RUNS, avg_time);

                //   
                final_result = b_copy; //   A_copy   ,   b_copy
            }

            //   
            std::cout << "\n\033[A\033[2K" << "Average execution time over " <<
                "\033[33m" << NUM_RUNS << "\033[0m"
                << " runs: " <<
                "\033[33m" << total_time / NUM_RUNS << "\033[0m"
                << " seconds." << std::endl;

            //  ,    Y
            if (display_choice == 'Y') {
                printResults(A, b, final_result);
            }
        }
    }

    return 0;
}
