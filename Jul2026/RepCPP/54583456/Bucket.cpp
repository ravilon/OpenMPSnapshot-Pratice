#include <algorithm>
#include <iostream>
#include <cstdio>
#include <vector>
#include <string>

#include "StringPair.h"
#include "StringOperations.h"
#include "Bucket.h"

using namespace std;

void Bucket::process(bool forceMerge){
	int oldNSegments;

	int mergeIndex = -1;
	int erasedIndex = -1;
	int* oldPositions = NULL;

	for(int i = 0; i < nSegments; i++){
		cout << "----------------------------\nSegmento "<< i << ":\n" << segments[i] << "\n";
	}
	///if(segments[0][segments[0].size()-1] != '.'){
	//	cout << "the last character in <full genome sequencing.> is not a point\n";
	//}
		string* oldSegments;
		int iteration = 1;
	//matriz com cada um dos segmentos comparado com os outros
	//a matriz não possui os elementos que estão na diagonal
	//nem os elementos acima da diagonal
	//(por motivos de economia de memoria)
		StringPair** oldMatrix = new StringPair*[nSegments];
		StringPair** matrix    = new StringPair*[nSegments];
		bool mergedSomething = false;
		while(nSegments > 1){

		#ifdef _OPENMP
			(void) omp_set_dynamic(0);
			(void) omp_set_num_threads(nSegments);
		#endif

		cout << "===============Iteration " << iteration << "=================\n";
		for (int i = 0; i < nSegments; i++){
			cout << "segments[" << i << "] = \"" 
			<< StringOperations::first20char(segments[i]) 
			<< " -> " 
			<< StringOperations::last20char(segments[i]) << "\"\n";
		}

		#pragma omp parallel
			{
			#pragma omp parallel for
				for(int length = 0; length < nSegments; length++){
					matrix[length]    = new StringPair[length];
					oldMatrix[length] = new StringPair[length];
				 	cout << "line " << length << " has " << length << " comparisons\n";
				}
			}

		//inicialização de cada comparação
		//ATENÇÃO: A iteração sobre a matriz deve obedecer este formato abaixo
		//	 		ou vão ocorrer falhas de segmentação
			int n = 0;
			int xMax = 1, yMax = 0;

			for(int i = 1; i < nSegments; i++){
				for(int j = 0; j < i; j++){
				//utilizar dados da matriz antiga
					bool reciclado = false;

					if((iteration > 1)
						&& (mergeIndex != -1 && erasedIndex != -1)
						&& (oldPositions != NULL)
						&& (j != mergeIndex && i != mergeIndex))
					{
						cout << "tentando reciclar " << i << " e " << j << " Passo 1" << endl;
						int oldI = oldPositions[i];
						int oldJ = oldPositions[j];
						if(j < i){
							cout << "tentando reciclar os antigos " << oldI << " e " << oldJ << " Passo 2" << endl;
							matrix[i][j] = oldMatrix[oldI][oldJ];
							reciclado = true;
							cout << "reciclado" << endl;
					}
				}
				if(!reciclado){
					matrix[i][j] = StringPair(segments[i],segments[j]);
					matrix[i][j].calcResult(forceMerge);
				}

				if(matrix[i][j].result.module > 0){
					cout << "Merging -" << segments[i] << "- with -" << segments[j] << "- results in ->\n" << matrix[i][j].result.result <<"\n";
				}
				if(matrix[i][j].result.module > matrix[xMax][yMax].result.module){
					xMax = i;
					yMax = j;
				}
				n++;
			}
		}


		//yMax sempre vem antes de xMax, a matriz garante isso!
		StringPair bestMerge = matrix[xMax][yMax];
		// cout << "segments " << yMax << " with " << xMax << ":\n";
		// cout << "--------------------\nThe best merge in the iteration is:\n ";
		// cout << "x = \"" << first20char(bestMerge.x) << " -> " << last20char(bestMerge.x) << "\"\n";
		// cout << "+\n";
		// cout << "y = \"" << first20char(bestMerge.y) << " -> " << last20char(bestMerge.y) << "\"\n";
		// cout << "(initial = " << bestMerge.result.initial << ")\n";
		// cout << "=\n";
		// cout << "(" << bestMerge.result.module << " em comum)\n";
		// cout << bestMerge.result.result << "\n";
		if(bestMerge.result.haveResult()){
			string* newSegments = new string[nSegments-1];
			oldPositions = new int[nSegments-1];
		
			for(int i = 0; i < yMax; i++){
				newSegments[i] = segments[i];
			}
			

			newSegments[yMax] = bestMerge.result.result;

			mergeIndex = yMax;
			erasedIndex  = xMax;

			for(int i = yMax+1; i < xMax; i++){
				newSegments[i] = segments[i];
			}
			

			for(int i = xMax+1; i < nSegments; i++){
				newSegments[i-1] = segments[i];
			}

			for(int i = 0; i < nSegments-1; i++){
				oldPositions[i] = i;
				if(i >= xMax){
					oldPositions[i]++;
				}
			}

			oldSegments = segments;
			segments = newSegments;

			oldNSegments = nSegments;
			nSegments -= 1;

			#pragma omp parallel
			{
				for(int i = 1; i < nSegments; i++){
					for(int j = 0; j < i; j++){
						oldMatrix[i][j] = matrix[i][j];
					}
				}
			}
		}else{
			break;
		}
		cout << "===============Iteration " << iteration << "=================\n\n";
		iteration++;
	}
	cout << "Final processing of the bucket[inside]\n";
	for (int i = 0; i < nSegments; i++){
		 cout << "segments[" << i << "] = \"" << segments[i] << "\"\n";
	}
}
