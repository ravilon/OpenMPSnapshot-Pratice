#ifndef CONV_HPP
#define CONV_HPP

#include <vector>
#include <queue>
#include <map>
#include <cstring>

// Sequential Algorithm
void filter_2d_inplace_seq(long long int n, long long int m, const std::vector<float>& K, std::vector<float>& A) {

    std::queue <float> buffer;
    int row_counter = 0;

    for(int i=0; i<n;i++){
        for(int j=0; j<m; j++){

            if(i-1<0 || i+1>=n || j-1<0 || j+1>=m){
                continue;
            }

            float ele_sum = 0.0;
            int conv_i, conv_j;

            // Convolution
            for (int k=0; k<9; k++){
                conv_i = i + (k / 3) - 1;
                conv_j = j + (k % 3) - 1;
                ele_sum += K[k]*A[conv_i*m+conv_j];
            }
            
            buffer.push(ele_sum);
        }

        row_counter++;

        
        if (row_counter==n-1){
            for(int buffer_i=0; buffer_i<buffer.size(); buffer_i++){
                int i_A =  (row_counter-2)*m+buffer_i;
                // To skip First and Last Columns
                if (buffer_i != 0 and buffer_i != m-1){
                    A[i_A] = buffer.front();
                    buffer.pop();
                }
            }
        }

        else if (row_counter>2){
            for(int buffer_i=0; buffer_i<m; buffer_i++){
                int i_A =  (row_counter-2)*m+buffer_i;
                // To skip First and Last Columns
                if (buffer_i != 0 and buffer_i != m-1){
                    A[i_A] = buffer.front();
                    buffer.pop();
                }
            }
            
        }

    }
} // filter_2d_inplace_seq

// Parallel Algorithm
void filter_2d_inplace_parallel(long long int n, long long int m, const std::vector<float>& K, std::vector<float>& A) {

    std::map<int, float> buffer;

    int row_counter = 0;

    for(int i=0; i<n;i++){

        #pragma omp parallel for default(none) shared(n,m,K,A,buffer,i)
        for(int j=0; j<m; j++){

            if(i-1<0 || i+1>=n || j-1<0 || j+1>=m){
                continue;
            }

            float ele_sum = 0.0;
            int conv_i, conv_j;

            // Convolution
            for (int k=0; k<9; k++){
                conv_i = i + (k / 3) - 1;
                conv_j = j + (k % 3) - 1;
                ele_sum += K[k]*A[conv_i*m+conv_j];
            }
            
            #pragma omp crtical
            buffer[i*m+j] = ele_sum;
        }
        

        row_counter++;
        

        if (i-1<0 || i+1>=n){
            continue;
        }

        /**if (row_counter==n-1){
            #pragma omp parallel for collapse(2) default(none) shared(n,m,A,row_counter,buffer)
            for(int buffer_i=row_counter-2; buffer_i<n; buffer_i++){
                for(int buffer_j=0; buffer_j<m;buffer_j++){
                    int idx_A =  buffer_i*m+buffer_j;

                    if (buffer_j != 0 && buffer_j != m-1 && buffer_i != 0 && buffer_i != n-1){
                        int idx_A =  buffer_i*m+buffer_j;
                        A[idx_A] = buffer[idx_A];
                        #pragma omp crtical
                        buffer.erase(idx_A);
                    }
                }
            }
        }

        else if (row_counter>2){
            #pragma omp parallel for default(none) shared(m,A,row_counter,buffer,i)
            for(int buffer_j=0; buffer_j<m; buffer_j++){
                int idx_A =  i*m+buffer_j;
                // To skip First and Last Columns
                if (buffer_j != 0 and buffer_j != m-1){
                    A[idx_A] = buffer[idx_A];
                    #pragma omp crtical
                    buffer.erase(idx_A);
                }
            }
            
        }**/

    }
} // filter_2d_inplace_parallel

// Sequential Algorithm
void filter_2d_seq(long long int n, long long int m, const std::vector<float>& K, std::vector<float>& A, std::vector<float>& C) {

    for(int i=0; i<n;i++){
        for(int j=0; j<m; j++){

            if(i-1<0 || i+1>=n || j-1<0 || j+1>=m){
                continue;
            }

            float ele_sum = 0.0;
            int conv_i, conv_j;

            // Convolution
            for (int k=0; k<9; k++){
                conv_i = i + (k / 3) - 1;
                conv_j = j + (k % 3) - 1;
                ele_sum += K[k]*A[conv_i*m+conv_j];
            }
            C[i*m+j] = ele_sum;
        }   

    }
} 
// filter_2d_seq

// Parallel Algorithm
void filter_2d_parallel(long long int n, long long int m, const std::vector<float>& K, std::vector<float>& A, std::vector<float>& C) {


    #pragma omp parallel for collapse(2) default(none) shared(n,m,A,K,C) 
    for(int i=0; i<n;i++){
        for(int j=0; j<m; j++){

            if(i-1<0 || i+1>=n || j-1<0 || j+1>=m){
                continue;
            }

            float ele_sum = 0.0;
            int conv_i, conv_j;

            // Convolution
            for (int k=0; k<9; k++){
                conv_i = i + (k / 3) - 1;
                conv_j = j + (k % 3) - 1;

                ele_sum += K[k]*A[conv_i*m+conv_j];
            }       
            C[i*m+j] = ele_sum;
        }   

    }
    
} 

#endif
