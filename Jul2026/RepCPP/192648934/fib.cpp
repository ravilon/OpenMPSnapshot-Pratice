/**
 * @brief  動的マルチスレッド化された手続きP-FIBの解析を行う
 * @date   2016/03/05
 */



//****************************************
// 必要なヘッダファイルのインクルード
//****************************************

#include <iostream>
#include <chrono>
#include <omp.h>



//****************************************
// マクロの定義
//****************************************

#define __PARALLEL__



//****************************************
// 型シノニム
//****************************************

using ll_t = long long;



//****************************************
// 関数の定義
//****************************************

/**
 * @brief  フィボナッチ数計算アルゴリズム
 * @param  n番目フィボナッチ数を求める
 * @return n版目のフィボナッチ数
 */
ll_t fib(ll_t n)
{
    if (n <= 1) {  // nが1以下のとき、再帰は底をつく
        return n;  // n(=1)を返す
    }
    else {
        ll_t x = fib(n - 1);
        ll_t y = fib(n - 2);

        return x + y;
    }
}



/**
 * @brief  動的マルチスレッド版フィボナッチ数計算アルゴリズム
 * @note   P-FIB(n)の並列度はT1(n)/T∞(n)=Θ(φ^n/n)であり、nが大きくなるにつれ急激に増加する
 * @param  n番目フィボナッチ数を求める
 * @return n版目のフィボナッチ数
 */
ll_t p_fib(ll_t n)
{
    ll_t x, y;
    if (n <= 1) {  // nが1以下のとき、再帰は底をつく
        return n;  // n(=1)を返す
    }
    else {
#pragma omp task shared(x) firstprivate(n)
        x = p_fib(n - 1);                   // x = spawn task p_fib(n-1)
#pragma omp task shared(y) firstprivate(n)
        y = p_fib(n - 2);                   // y = spawn task p_fib(n-2)
#pragma omp taskwait
        return x + y;                       // sync return x + y
    }
}

int main()
{

    int n = 20;
    int fibn;
#ifdef __PARALLEL__
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    // 並列版フィボナッチ計算
    auto start = std::chrono::system_clock::now();
#pragma omp parallel shared(n)
    {
        #pragma omp single
        fibn = p_fib(n);
    }
    auto end = std::chrono::system_clock::now();
    auto dur = end - start;
    auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
    std::cout << "parallel fib(" << n << ")=" << fibn << std::endl;
    std::cout << msec << " milli sec\n";

#else
    // 逐次版フィボナッチ計算
    auto start = std::chrono::system_clock::now();
    fibn = fib(n);;
    auto end = std::chrono::system_clock::now();
    auto dur = end - start;
    auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
    std::cout << "fib(" << n << ")=" << fibn << std::endl;
    std::cout << msec << " milli sec\n";
#endif
    
    getchar();
    return 0;
}

