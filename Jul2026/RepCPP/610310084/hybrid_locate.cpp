#include "dcel.h"
#include "vertex.h"
#include "halfedge.h"
#include "face.h"
#include <limits>

void DCEL::hybrid_locations(std::vector <pt> &points, std::vector <int> &out)
{
    int n = points.size();
    #pragma omp parallel shared(out, points)
    #pragma omp single
    for (int i = 0; i < n; i++)
    {
	#pragma omp task
	sequential_locate(points[i], out[i]);
    }
    #pragma omp taskwait
}
