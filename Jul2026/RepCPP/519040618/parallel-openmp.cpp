#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

typedef unsigned short mtype;

using std::vector;
using std::string;

// leitura de arquivos para string
string read_seq(string file) {
	std::ifstream t(file);
	std::stringstream buffer;
	while(buffer << t.rdbuf());
	return buffer.str();
}

// string de letras únicas a partir da string 'str'
void buildUniqueLetters(string str, string &unique){
	for(int i = 0; i < str.length(); ++i){
		// checa se a letra já está no vetor de letras únicas
		if(find(unique.begin(), unique.end(), str[i]) == unique.end())
			unique.push_back(str[i]);
	}
}

// acha o índice da primeira letra igual ao caracter 'c' na string 'str'
int firstEqual(string str, char c){
	for(int i = 0; i < str.length(); ++i){
		if(str[i] == c) return i;
	}
	return -1;
}

// processa a matriz P, a matriz que contém quando determinado caracter de 'bString' foi
// visto por último, utilizado para tirar a dependência da recorrência principal
void calculatePMatrix(string unique, string bString, vector<vector<mtype>> &p){
	#pragma omp parallel for
	for(int i = 0; i < unique.length(); ++i){
		for(int j = 1; j < bString.length()+1; ++j){
			if(bString[j-1] == unique[i]){
				p[i][j] = j;
			} else {
				p[i][j] = p[i][j-1];
			}
		}
	}
}

// calcula o score da maior subsequencia em comum
mtype LCS(vector<vector<mtype>> &scoreMatrix, string a, std:: string b, vector<vector<mtype>> p, string unique) {
	for (int i = 1; i < a.length()+1; i++) {
		mtype c = firstEqual(unique, a[i-1]);
		#pragma omp parallel for
		for (int j = 0; j < b.length()+1; j++) {
			// aplica a programação dinâmica utilizando como base a recursão e 
			// também utilizando a matriz P
			if(i == 0 || j == 0){
				scoreMatrix[i][j] = 0;
			} else if(p[c][j] == 0){
				scoreMatrix[i][j] = max(scoreMatrix[i-1][j], 0);
			} else {
				scoreMatrix[i][j] = max(scoreMatrix[i-1][j], scoreMatrix[i-1][p[c][j]-1] + 1);
			}
		}
	}
	return scoreMatrix[a.length()][b.length()];
}

void printMatrix(int n, int m, vector<vector<mtype>> &mat){
	for(mtype i = 0; i < n; ++i){
	 	for(mtype j = 0; j < m; ++j){
	 		std::cout << mat[i][j] << ' ';
		}
	 	std::cout << std::endl;
	}
	std::cout << std::endl;
}

int main(int argc, char ** argv) {
	if(argc < 3){
		fprintf(stderr, "Necessário arquivos de entrada como argumento\nUso: ./paralelo A.in B.in\n");
		exit(1);
	}	

	string seqA, seqB;
	
	string unique;

	seqA = read_seq(argv[1]);
	seqB = read_seq(argv[2]);

	buildUniqueLetters(seqA, unique);
	buildUniqueLetters(seqB, unique);

	vector<vector<mtype>> scoreMatrix (seqA.length()+1, vector<mtype>(seqB.length()+1));

	vector<vector<mtype>> p (unique.length(), vector<mtype>(seqB.length()+1));

	calculatePMatrix(unique, seqB, p);

	mtype score = LCS(scoreMatrix, seqA, seqB, p, unique);

	//printMatrix(seqA.length(), seqB.length(), scoreMatrix);

	//printMatrix(unique.length(), seqB.length(), p);

	std::cout << "\nScore: " << score << std::endl;

	return EXIT_SUCCESS;
}
