#include "dcel.h"
#include "vertex.h"
#include "halfedge.h"
#include "face.h"
#include <limits>

const float EPS = 1e-5;
const ptT INF = 1e20;

distance_index_pair pair_min(distance_index_pair a, distance_index_pair b)
{
    return (a.first < b.first) ? a : b;
}

face* DCEL::parallel_get_closest_face(pt &point)
{
    distance_index_pair closest(1e20, -1);
    distance_index_pair pairs[numSample];

    #pragma omp declare reduction (reduction_min_pair : distance_index_pair : omp_out=pair_min(omp_out, omp_in)) initializer(omp_priv = distance_index_pair(1e20, -1))
    #pragma omp parallel
    {
	#pragma omp for
	for (int i = 0; i < numSample; i++)
	    pairs[i] = {pt::distance2(samplefaces[i] -> centroid, point), i};

        #pragma omp for reduction(reduction_min_pair : closest)
	for (int i = 0; i < numSample; i++)
	{
	    closest = pair_min(closest, pairs[i]);
	}
    }
    
    return samplefaces[closest.second];
}

void parallel_locate_recursive(face* curr_face, pt &point, int &result)
{
    face* next_face = nullptr;
    std::pair <pt, pt> ray = {curr_face -> centroid, point};
    auto edges = curr_face -> getEdges();
    for (halfedge* e: edges)
    {
	auto endpoints = e -> getEndpoints();
	if (pt::intersects(ray.first, ray.second, endpoints.first, endpoints.second))
	{
	    next_face = e -> twin -> incidentFace;
	}
    }
    if (!next_face)
    {
	result = curr_face -> index;
    }
    else if (next_face -> index < 0)
    {
	result = -1;
    }
    else
    {
	parallel_locate_recursive(next_face, point, result);
    }
}

void DCEL::parallel_locate(pt &point, int &result)
{
    face* start_face = parallel_get_closest_face(point);
    parallel_locate_recursive(start_face, point, result);
}

void DCEL::parallel_locations(std::vector <pt> &points, std::vector <int> &out)
{
    int n = points.size();
    #pragma omp parallel for shared(out, points)
    for (int i = 0; i < n; i++)
    {
	#pragma omp task
	parallel_locate(points[i], out[i]);
    }
    #pragma omp taskwait
}
