#pragma once
#include <cstdint>
#include <limits>
#include <utility>
#include <random>
#include <atomic>
#include <array>

using std::array;
using std::atomic;
using std::numeric_limits;
using std::uniform_int_distribution;




// PARITY1
// -------
// Find 1-bit parity of a value.

int parity1(uint32_t x) {
  x ^= x >> 16;
  x ^= x >> 8;
  x ^= x >> 4;
  x &= 0xF;
  return (0x6996 >> x) & 1;
}

int parity1(uint64_t x) {
  return parity1(uint32_t(x)) ^ parity1(uint32_t(x >> 32));
}

int parity1(uint16_t x) {
  return parity1(uint32_t(x));
}

template <class T>
int parity1(const T *x, size_t N) {
  int a = 0;
  for (size_t i=0; i<N; ++i)
    a ^= parity1(x[i]);
  return a;
}

template <class T, size_t N>
int parity1(array<T, N> x) {
  int a = 0;
  for (T v : x)
    a ^= parity1(v);
  return a;
}

template <class T, size_t N>
int parity1(const atomic<array<T, N>>& x) {
  return parity1(x.load());
}




// RANDOM WITH PARITY1
// -------------------

template <class R, class T>
T randomWithParity1(const T& _, R& rnd, int p) {
  uniform_int_distribution<T> valDis(0, numeric_limits<T>::max());
  uniform_int_distribution<>  bitDis(0, numeric_limits<T>::digits);
  T a = valDis(rnd);
  return parity1(a)==p? a : a ^ (1 << bitDis(rnd));
}

template <class R, class T>
void randomWithParity1W(T *a, size_t N, R& rnd, int p) {
  int q = 0;
  uniform_int_distribution<T> valDis(0, numeric_limits<T>::max());
  uniform_int_distribution<>  idxDis(0, N-1);
  for (size_t i=0; i<N; ++i) {
    a[i] = valDis(rnd);
    q   ^= parity1(a[i]);
  }
  if (q!=p) a[idxDis(rnd)] ^= 1;
}

template <class R, class T, size_t N>
array<T, N> randomWithParity1(const array<T, N>& _, R& rnd, int p) {
  array<T, N> a; randomWithParity1W(a.data(), N, rnd, p);
  return a;
}




// RANDOM WITH SPLIT ODD PARITY1
// -----------------------------

template <class R>
uint32_t randomWithSplitOddParity1(const uint32_t& _, R& rnd) {
  uint32_t l = randomWithParity1(uint16_t(), rnd, 1);
  uint32_t h = randomWithParity1(uint16_t(), rnd, 1);
  return (h << 16) | l;
}

template <class R>
uint64_t randomWithSplitOddParity1(const uint64_t& _, R& rnd) {
  uint64_t l = randomWithParity1(uint32_t(), rnd, 1);
  uint64_t h = randomWithParity1(uint32_t(), rnd, 1);
  return (h << 32) | l;
}

template <class R, class T>
void randomWithSplitOddParity1W(T *a, size_t N, R& rnd) {
  randomWithParity1W(a, N/2, rnd, 1);
  randomWithParity1W(a+ N/2, N - N/2, rnd, 1);
}

template <class R, class T, size_t N>
array<T, N> randomWithSplitOddParity1(const array<T, N>& _, R& rnd) {
  array<T, N> a; randomWithSplitOddParity1W(a.data(), N, rnd);
  return a;
}

template <class R, class T, size_t N>
array<T, N> randomWithSplitOddParity1(const atomic<array<T, N>>& _, R& rnd) {
  array<T, N> a; randomWithSplitOddParity1W(a.data(), N, rnd);
  return a;
}
