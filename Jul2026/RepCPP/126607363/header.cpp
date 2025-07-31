#include <cstdio>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <string>
#include <limits>
#include <omp.h>

using namespace std;

#define CACHE_LINE_SIZE 16392
// #define INPUT_EXP 11
#define INPUT_EXP 11

size_t cnt = 0;
size_t ins = 0;

#ifdef INSTR
	inline void INS(size_t cnt) {
		#pragma omp atomic 
		ins += (cnt);
	}
#else
	inline void INS(size_t cnt) {
	}
#endif

#ifdef DEBUG
	#define KMP_RES(index) cout << "KMP: pattern found at index: " << (index) << endl;
	#define NAIVE_RES(index) cout << "Naive: pattern found at index: " << (index) << endl;
#else
	#define KMP_RES(index)
	#define NAIVE_RES(index)
#endif
