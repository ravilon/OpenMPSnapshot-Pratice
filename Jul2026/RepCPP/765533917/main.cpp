#include "gpu_mul.h"
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <chrono>
#include <omp.h>
#include <immintrin.h>


void cpuMatrixMultiplyNaive(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b, std::vector<std::vector<float>>& c);
void cpuMatrixMultiplyOpenMP(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b, std::vector<std::vector<float>>& c);
void cpuMatrixMultiplyAVX2(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b, std::vector<std::vector<float>>& c);
void cpuMatrixMultiplyOpenMPAVX(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b, std::vector<std::vector<float>>& c);

void cpuMatrixMultiplyNaive(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b, std::vector<std::vector<float>>& c) {
    int N = a.size();
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            c[row][col] = 0;
            for (int k = 0; k < N; ++k) {
                c[row][col] += a[row][k] * b[k][col];
            }
        }
    }
}

void cpuMatrixMultiplyOpenMP(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b, std::vector<std::vector<float>>& c) {
    int N = a.size();
    int row, col, k;
    #pragma omp parallel for shared(a, b, c) private(row, col, k)
    for (row = 0; row < N; ++row) {
        for (col = 0; col < N; ++col) {
            c[row][col] = 0;
            for (k = 0; k < N; ++k) {
                c[row][col] += a[row][k] * b[k][col];
            }
        }
    }
}
// 128 비트 벡터 연산
void cpuMatrixMultiplySSE2(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b, std::vector<std::vector<float>>& c) {
    int N = a.size();

    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            __m128 sum = _mm_setzero_ps();
            for (int k = 0; k < N; k += 4) {
                __m128 aVec = _mm_loadu_ps(&a[row][k]);
                __m128 bVec = _mm_set_ps(b[k+3][col], b[k+2][col], b[k+1][col], b[k][col]);

                sum = _mm_add_ps(sum, _mm_mul_ps(aVec, bVec));
            }
            sum = _mm_hadd_ps(sum, sum);
            sum = _mm_hadd_ps(sum, sum);
            c[row][col] = _mm_cvtss_f32(sum);
        }
    }
}

// 256 비트 벡터 연산
void cpuMatrixMultiplyAVX2(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b, std::vector<std::vector<float>>& c) {
    int N = a.size();

    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            __m256 sum = _mm256_setzero_ps();
            for (int k = 0; k < N; k += 8) {
                __m256 aVec = _mm256_loadu_ps(&a[row][k]);
                __m256 bVec = _mm256_set_ps(b[k+7][col], b[k+6][col], b[k+5][col], b[k+4][col], 
                                            b[k+3][col], b[k+2][col], b[k+1][col], b[k][col]);

                sum = _mm256_add_ps(sum, _mm256_mul_ps(aVec, bVec));
            }
            sum = _mm256_hadd_ps(sum, sum);
            sum = _mm256_hadd_ps(sum, sum);
            float temp[8];
            _mm256_storeu_ps(temp, sum);
            c[row][col] = temp[0] + temp[6];
        }
    }


}

void cpuMatrixMultiplyAVX2OPENMP(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b, std::vector<std::vector<float>>& c) {
    int N = a.size();
    #pragma omp parallel for
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            __m256 sum = _mm256_setzero_ps();
            for (int k = 0; k < N; k += 8) {
                __m256 aVec = _mm256_loadu_ps(&a[row][k]);
                __m256 bVec = _mm256_set_ps(b[k+7][col], b[k+6][col], b[k+5][col], b[k+4][col], 
                                            b[k+3][col], b[k+2][col], b[k+1][col], b[k][col]);

                sum = _mm256_add_ps(sum, _mm256_mul_ps(aVec, bVec));
            }
            sum = _mm256_hadd_ps(sum, sum);
            sum = _mm256_hadd_ps(sum, sum);
            float temp[8];
            _mm256_storeu_ps(temp, sum);
            c[row][col] = temp[0] + temp[6];
        }
    }
}


int main(int argc, char** argv) {

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " N <mode>" << std::endl;
        std::cerr << "Modes: naive, openmp, avx, openmp_avx" << std::endl;
        return 1;
    }

    int N = std::stoi(std::string(argv[1]));
    std::string mode = argv[2];


    if (mode == "naive") {
    } 
    else if (mode == "openmp") {
    } 
    else if (mode == "avx") {
    } 
    else if (mode == "openmp_avx") {
    } 
    else if (mode == "sse") {
    } 
    else if (mode == "avxop") {

    }
    else {
        std::cerr << "Invalid mode. Available modes: naive, openmp, avx, openmp_avx" << std::endl;
        return 1;
    }


    std::random_device rd;
    std::mt19937 eng(rd());
    std::uniform_real_distribution<> distr(0.0, 1.0); 
    
    std::vector<std::vector<float>> a(N, std::vector<float>(N, 1.0f));
    std::vector<std::vector<float>> b(N, std::vector<float>(N, 2.0f));
    
    
    
    std::vector<std::vector<float>> c(N, std::vector<float>(N, 0.0f));
    std::cout << "# of processors : " << omp_get_num_procs() << std::endl;
    omp_set_num_threads(omp_get_num_procs());

    auto start_cpu = std::chrono::high_resolution_clock::now();
    if (mode == "naive")
        for (int i=0;i<100;i++)
        {
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    a[i][j] = distr(eng);
                    b[i][j] = distr(eng);
                }
            }
            cpuMatrixMultiplyNaive(a, b, c);
        }
    else if (mode == "openmp")
        for (int i=0;i<100;i++)
        {
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    a[i][j] = distr(eng);
                    b[i][j] = distr(eng);
                }
            }
            cpuMatrixMultiplyOpenMP(a, b, c);
        }
    else if (mode == "avx")
         for (int i=0;i<100;i++)
         {
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    a[i][j] = distr(eng);
                    b[i][j] = distr(eng);
                }
            }
            cpuMatrixMultiplyAVX2(a, b, c);
         }
    else if (mode == "sse")
         for (int i=0;i<100;i++)
         {
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    a[i][j] = distr(eng);
                    b[i][j] = distr(eng);
                }
            }
            cpuMatrixMultiplySSE2(a, b, c);
         }
    else if (mode == "openmp_avx")
         for (int i=0;i<100;i++){
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    a[i][j] = distr(eng);
                    b[i][j] = distr(eng);
                }
            }
            cpuMatrixMultiplyAVX2OPENMP(a, b, c);
         }
    else
    {
        return -1;
    }

    auto end_cpu = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> duration_cpu = end_cpu - start_cpu;
    std::cout << "CPU Execution Time: " << duration_cpu.count() << " ms" << std::endl;

    std::cout << "CPU Result:" << std::endl;
    // for (const auto &row : c) {
    //     for (float elem : row) {
    //         std::cout << elem << " ";
    //     }
    //     std::cout << std::endl;
    // }

    

    auto start_gpu = std::chrono::high_resolution_clock::now();
    for (int i=0;i<100;i++){
        float *gpu_a = new float[N*N];
        float *gpu_b = new float[N*N];
        float *gpu_c = new float[N*N];
        for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    a[i][j] = distr(eng);
                    b[i][j] = distr(eng);
                }
        }
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                gpu_a[i*N + j] = a[i][j];
                gpu_b[i*N + j] = b[i][j];
            }
        }
        gpuMatrixMultiply(gpu_a, gpu_b, gpu_c, N);
        delete[] gpu_a;
        delete[] gpu_b;
        delete[] gpu_c;
    }
    
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration_gpu = end_gpu - start_gpu;
    std::cout << "GPU Execution Time: " << duration_gpu.count() << " ms" << std::endl;    

    start_gpu = std::chrono::high_resolution_clock::now();
    for (int i=0;i<100;i++){
        float *gpu_a = new float[N*N];
        float *gpu_b = new float[N*N];
        float *gpu_c = new float[N*N];
        for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    a[i][j] = distr(eng);
                    b[i][j] = distr(eng);
                }
        }
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                gpu_a[i*N + j] = a[i][j];
                gpu_b[i*N + j] = b[i][j];
            }
        }
        gpuMatrixMultiplyShared(gpu_a, gpu_b, gpu_c, N);
        delete[] gpu_a;
        delete[] gpu_b;
        delete[] gpu_c;
    }
    end_gpu = std::chrono::high_resolution_clock::now();
    duration_gpu = end_gpu - start_gpu;
    std::cout << "GPU Execution Time using Shared 64: " << duration_gpu.count() << " ms" << std::endl; 
    std::cout << "GPU Result:" << std::endl;
    // for (int i = 0; i < N; ++i) {
    //     for (int j = 0; j < N; ++j) {
    //         std::cout << gpu_c[i*N + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    return 0;
}
