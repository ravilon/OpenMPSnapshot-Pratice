/*
 * g++ -std=c++14 InvOrderBits.cpp -fopenmp -O3 -o InvOrderBits
 */

#include <iostream>
#include <bitset>
#include <chrono>
#include <omp.h>

using namespace std::chrono;

// Macros to generate the lookup table (at compile-time)
#define R2(n) n, n + 2*64, n + 1*64, n + 3*64
#define R4(n) R2(n), R2(n + 2*16), R2(n + 1*16), R2(n + 3*16)
#define R6(n) R4(n), R4(n + 2*4 ), R4(n + 1*4 ), R4(n + 3*4 )
#define REVERSE_BITS R6(0), R6(2), R6(1), R6(3)

// lookup table to store the reverse of each index of the table.
// The macro `REVERSE_BITS` generates the table
uint64_t lookup[256] = {REVERSE_BITS};

// Function to reverse bits of `n` using a lookup table
uint64_t reverseBits(uint64_t n) {

    /* Assuming a 64–bit (8 byt380296es) integer, break the integer into 8–bit chunks.
       Note: mask used `0xff` is `11111111` in binary */

    uint64_t reverse = lookup[n & 0xff] << 56 |                // consider the first 8 bits
                       lookup[(n >> 8) & 0xff] << 48 |         // consider the next 8 bits
                       lookup[(n >> 16) & 0xff] << 40 |        // consider the next 8 bits
                       lookup[(n >> 24) & 0xff] << 32 |        // consider the next 8 bits
                       lookup[(n >> 32) & 0xff] << 24 |        // consider the next 8 bits
                       lookup[(n >> 40) & 0xff] << 16 |        // consider the next 8 bits
                       lookup[(n >> 48) & 0xff] << 8 |         // consider the next 8 bits
                       lookup[(n >> 56) & 0xff];               // consider last 8 bits

    return reverse;
}

int main() {
    // data setup
    uint64_t *input_values;
    //input_values = (uint64_t *) malloc(sizeof(uint64_t) * 100000000);
    input_values = (uint64_t *) aligned_alloc(sizeof(uint64_t), sizeof(uint64_t) * 100000000);
    uint64_t *output_values;
    //output_values = (uint64_t *) malloc(sizeof(uint64_t) * 100000000);
    output_values = (uint64_t *) aligned_alloc(sizeof(uint64_t), sizeof(uint64_t) * 100000000);

    for (int i = 0; i < 100000000; i++) {
        input_values[i] = i;
    }

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for default(shared)
    for (int i = 0; i < 100000000; i++) {
        output_values[i] = reverseBits(input_values[i]);
    }

    auto stop = std::chrono::high_resolution_clock::now();

    // check the result
    std::bitset<64> x(input_values[26]);
    std::cout << x << '\n';

    std::bitset<64> y(output_values[26]);
    std::cout << y << '\n';

    std::cout << "Done in " << duration_cast<std::chrono::microseconds>(stop - start).count() << " us" << std::endl;
    return 0;
}
