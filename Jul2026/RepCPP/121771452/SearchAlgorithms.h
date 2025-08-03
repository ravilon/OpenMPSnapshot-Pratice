#pragma once

#include <vector>
#include <omp.h>
#include <array>

namespace SearchAlgorithms
{

constexpr unsigned MAX_THREAD_COUNT = 64;
using Shbool = short;	// my reference-able bool

/*
val = 7
sortedVec = 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14

15 elements divided to 4 threads => segmSize = 3

Each thread checks if "val" is to their right
            T0       T1       T2       T3                RB (Right-Border)
sortedVec = 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14
isValRght = V        V        V        X                 X

T2 sees that val is to his right and T3 says it's not, so it shrinks the search interval
=> l = 6, r = 8; => segmSize = 0 => exit while loop

            T0 T1 T2
sortedVec = 6  7  8

=> T1 will find the value at index 7
*/
template<typename VecType, typename VecSizeT = std::vector<VecType>::size_type>
VecSizeT ShorterParallelSearch(const std::vector<VecType>& sortedVec, const VecType& val)
{
	const unsigned threadCount = omp_get_max_threads();
	const unsigned lastThrdID = threadCount - 1;

	// each thread will tell if "val" is to their "right"
	std::array<Shbool, MAX_THREAD_COUNT+1> isValToTheRight; // + 1 <=> the right border which is just queried by the last thread
	isValToTheRight[threadCount] = false;	// the right border will say "false" every time

	VecSizeT l = 0;	// left most index
	VecSizeT r = sortedVec.size() - 1; // right most index
	VecSizeT indexFound = -1;

	#pragma omp parallel \
		shared(sortedVec, val, l, r, isValToTheRight, indexFound) \
		default(none)	// compile error if a local variable is not mentioned in shared(...) or private(...)
	{
		const unsigned thrdID = omp_get_thread_num();

		Shbool& currThrdSaysItsRight = isValToTheRight[thrdID];
		const Shbool& nextThrdSaysItsRight = isValToTheRight[thrdID + 1];

		while (l <= r - threadCount)
		{
			const VecSizeT segmSize = (r - l + 1) / (threadCount + 1) + 1;
			const VecSizeT currThrdPos = l + segmSize * thrdID;
			currThrdSaysItsRight = (sortedVec[currThrdPos] <= val);

			// waiting for every thread to finish checking if "val" is on their right
			#pragma omp barrier

			if (currThrdSaysItsRight != nextThrdSaysItsRight)
			{
				l = currThrdPos;
				if (thrdID != lastThrdID)
					r = currThrdPos + segmSize - 1;
			}

			#pragma omp barrier
		}

		const VecSizeT currThrdPos = l + thrdID;
		if (currThrdPos <= r && sortedVec[currThrdPos] == val)
			indexFound = currThrdPos;
	}

	return indexFound;
}

/* ! In this version, threads look left and there is 1 more segment
val = 8
sortedVec = 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17

18 elements divided to (4 threads + 1) segments => segmSize = 3 + 1 = 4

Each thread checks if "val" is to their right and then check the result with their previous thread
         LB(LeftBorder)  T0          T1          T2          T3
sortedVec =     0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17
isValRght = V            V           V           X           X
                                     ^           ^
					             these 2 are different
after the first while:
             LB T0 T1 T2 T3
sortedVec = ... 7  8  9  10 ...
isValRght =  V  V  V  X  X

=> T2 modifies l = 8, r = 8 so "val" must be at the index 8
*/
template<typename VecType, typename VecSizeT = std::vector<VecType>::size_type>
VecSizeT ParallelSearch(const std::vector<VecType>& sortedVec, const VecType& val)
{
	const unsigned threadCount = omp_get_max_threads();
	const unsigned firstThrdID = 0;

	// each thread will tell if "val" is to their "right"
	std::array<Shbool, MAX_THREAD_COUNT + 1> isValToTheRight; // + 1 <=> the left border
	isValToTheRight[0] = true;	// the left border will say "true" every time

	std::array<VecSizeT, MAX_THREAD_COUNT> thrdsPos;	// position for each thread

	VecSizeT l = 0;	// left most index
	VecSizeT r = sortedVec.size() - 1; // right most index

	#pragma omp parallel \
		shared(l, r, thrdsPos, isValToTheRight, sortedVec, val) \
		default(none)	// compile error if a local variable is not mentioned in shared(...) or private(...)
	{
		const unsigned thrdID = omp_get_thread_num();
		VecSizeT& currThrdPos = thrdsPos[thrdID];
		Shbool& currThrdSaysItsRight = isValToTheRight[thrdID + 1];
		const Shbool& prevThrdSaysItsRight = isValToTheRight[thrdID];

		while (l < r)
		{
			const VecSizeT segmSize = (r - l + 1) / (threadCount + 1) + 1;
			currThrdPos = l + segmSize * (thrdID + 1) - 1;
			currThrdSaysItsRight = (sortedVec[currThrdPos] <= val);

			#pragma omp barrier

			if (currThrdSaysItsRight != prevThrdSaysItsRight)
			{
				r = currThrdPos - 1;
				if (thrdID != firstThrdID)
				{
					l = thrdsPos[thrdID - 1];
				}
			}

			#pragma omp single
			{
				// no thread looked at this segment so one needs to check whether val is in the last segment
				if (isValToTheRight[threadCount])
				{
					l = thrdsPos[threadCount-1];
				}
			}
		}
	}

	// now l == r

	// if "val" should've been here but it isn't
	if (sortedVec[l] != val)
		return VecSizeT(-1);

	return l;
}

template<typename VecType, typename VecSizeT = std::vector<VecType>::size_type>
VecSizeT BinarySearch(const std::vector<VecType>& sortedVec, VecType val)
{
	VecSizeT l = 0, r = sortedVec.size() - 1, m;
	while (l < r)
	{
		m = l + (r - l) / 2 + 1;
		if (sortedVec[m] <= val)
		{
			l = m;
		}
		else
		{
			r = m - 1;
		}
	}

	// now l == r

	// if "val" should've been here but it isn't
	if (sortedVec[l] != val)
		return VecSizeT(-1);

	return l;
}

template<typename VecType, typename VecSizeT = std::vector<VecType>::size_type>
VecSizeT BinarySearchInParallel(const std::vector<VecType>& sortedVec, const VecType& val)
{
	VecSizeT idxFound = -1;
	const unsigned threadCount = omp_get_max_threads();
	const VecSizeT vecSize = sortedVec.size();

	if (vecSize <= threadCount)
	{
		#pragma omp parallel shared(idxFound)
		{
			const unsigned thrdID = omp_get_thread_num();
			if (thrdID < vecSize && sortedVec[thrdID] == val)
				idxFound = thrdID;
		}

		return idxFound;
	}

	#pragma omp parallel
	{
		const unsigned thrdID = omp_get_thread_num();
		VecSizeT segmSize = vecSize / threadCount;
		VecSizeT l = segmSize * thrdID;
		VecSizeT r = segmSize * (thrdID + 1);

		if (thrdID == threadCount - 1)	// last thread gets a little more work
		{
			segmSize += vecSize % threadCount;
			l = vecSize - segmSize;
			r = vecSize - 1;
		}

		while (l < r)
		{
			const VecSizeT m = l + (r - l) / 2 + 1;
			if (sortedVec[m] <= val)
			{
				l = m;
			}
			else
			{
				r = m - 1;
			}
		}

		if (sortedVec[l] == val)
			idxFound = l;
	}
	
	return idxFound;
}

}

