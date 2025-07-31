/**
 * @brief  マルチスレッドアルゴリズムにおける決定性競合(determinacy race)の例を発生させる
 * @date   2016/03/05
 */



//****************************************
// 必要なヘッダファイルのインクルード
//****************************************

#include <iostream>
#include <chrono>
#include <omp.h>
#include "../../../MiLib/Random/xorshift.hpp"



//****************************************
// 型シノニム
//****************************************

using elem_t = std::int32_t;
using vec_t = std::vector<elem_t>;
using mat_t = std::vector<vec_t>;




//****************************************
// 関数の定義
//****************************************

/**
 * @brief  ベクトルと行列の乗算の間違った実現
 * @param  const mat_t& A nxn型行列A
 * @param  const vec_t& x nベクトルx
 * @return vec_t y
 */
vec_t matvec_wrong(const mat_t& A, const vec_t& x)
{
    int i, j;
    
    std::size_t n = A.size();  // n = A.rows
    vec_t y(n);                // yを長さnの新たなベクトルとする
#pragma omp parallel for private(i)
    for (i = 0; i < n; i++) {  // 並列に初期化
        y[i] = 0;
    }

    // メインループ
#pragma omp parallel
    {
#pragma omp task private(i)
        for (i = 0; i < n; i++) {
#pragma omp task private(j)
            for (j = 0; j < n; j++) {
                y[i] = y[i] + A[i][j] * x[j];  // ここで競合
            }
        }
    }
    return y;
}


int main()
{
    const int N = 4;

    vec_t x(N);
    mat_t A(N, vec_t(N));
    XorShiftRNG rng;
    
    std::cout << "A = ["; 
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = 1;
            std:: cout << A[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "]" << std::endl;

    std::cout << "x = [";
    for (int i = 0; i < N; i++) {
        x[i] = 1;
        std::cout << x[i] << " ";
    }
    std::cout << "]" << std::endl;


    omp_set_dynamic(0);
    omp_set_num_threads(4);
    omp_set_nested(true);


     vec_t y;
#pragma omp parallel shared(A, x)
    {
        #pragma omp single
         y = matvec_wrong(A, x);
    }

    std::cout << "y = [";
    for (int i = 0; i < N; i++) {
        std::cout << y[i] << " ";
    }
    std::cout << "]" << std::endl;
    

    return 0;
}

