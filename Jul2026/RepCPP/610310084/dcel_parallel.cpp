#include "dcel.h"
#include "vertex.h"
#include "halfedge.h"
#include "face.h"
#include "sort.h"
#include <assert.h>

void DCEL::parallel_make_DCEL(std::vector <pt> &points, std::vector <std::vector<int>> &triangles)
{
    assert(vertices.empty() and faces.empty() and edges.empty());
    
    parallel_createPointSet(points);
    std::vector <endpoint_indices> created_edges; // Used to detect twin half-edges after sorting
    parallel_buildFaces(triangles, created_edges);
    std::vector <endpoint_indices> twinless_edges; // Boundary edges are half-edges without twins
    parallel_matchTwinEdges(created_edges, twinless_edges);
    parallel_buildExterior(twinless_edges);

    assert(vertices.size() == points.size());
}

void DCEL::parallel_createPointSet(std::vector <pt> &points)
{
    vertices.resize(points.size());
    #pragma omp parallel for
    for (int i = 0; i < points.size(); i++)
    {
	vertices[i] = new vertex(points[i], i);
    }
}

void DCEL::parallel_buildFaces(std::vector <std::vector<int>> &triangles, std::vector <endpoint_indices> &created_edges)
{
    faces.resize(triangles.size());
    edges.resize(3 * triangles.size());
    created_edges.resize(3 * triangles.size());
    #pragma omp parallel for
    for (int i = 0; i < triangles.size(); i++)
    {
	int indices[3] = {triangles[i][0], triangles[i][1], triangles[i][2]};

	auto orient = pt::orientation(vertices[indices[0]] -> point, vertices[indices[1]] -> point, vertices[indices[2]] -> point);
	if (orient > 0) // Ensure that half edges in a face are ccw oriented
	{
	    std::swap(indices[0], indices[2]);
	}

	faces[i] = new face(i);
	face* curr_face = faces[i];
	
	halfedge* triangle_edges[3]; // j-th half edge originates from indices[j]
	for (int j = 0; j < 3; j++)
	{
	    triangle_edges[j] = new halfedge;
	    edges[3 * i + j] = triangle_edges[j];
	}

	for (int j = 0; j < 3; j++)
	{
	    int jnext = (j + 1) % 3;

	    triangle_edges[j] -> setIncidentFace(curr_face); // Every edge belongs to same face
	    triangle_edges[j] -> setOrigin(vertices[indices[j]]);// Set origin of halfedge and incidentEdge for origin
	    triangle_edges[j] -> setNext(triangle_edges[jnext]); // Set next edge for face traversal

	    created_edges[3 * i + j] = endpoint_indices(triangles[i][j], triangles[i][jnext], edges[3 * i + j]);
	}
    }
}

void DCEL::parallel_matchTwinEdges(std::vector <endpoint_indices> &created_edges, std::vector <endpoint_indices> &twinless_edges)
{
    auto radix_sort = [](std::vector <endpoint_indices> &vp, int numPoints)
		      {
			  auto getSecond = [](endpoint_indices vp){return vp.indices.second;};
			  bucket_sort(vp, numPoints, getSecond);
			  auto getFirst = [](endpoint_indices vp){return vp.indices.first;};
			  bucket_sort(vp, numPoints, getFirst);
		      };
    radix_sort(created_edges, vertices.size());
    
    for (int i = 0; i < created_edges.size(); )
    {
	if (i + 1 < created_edges.size() and created_edges[i].indices == created_edges[i + 1].indices)
	{
	    created_edges[i].edge -> setTwin(created_edges[i + 1].edge);
	    i += 2;
	}
	else
	{
	    twinless_edges.push_back(created_edges[i]);
	    i++;
	}
    }
}

void DCEL::parallel_buildExterior(std::vector <endpoint_indices> &twinless_edges)
{
    exterior = getNewFace(-1);

    std::vector <halfedge*> exterior_edges(vertices.size()); // Used to connect exterior edges together

    for (int i = 0; i < twinless_edges.size(); i++)
    {
	halfedge* reverse_edge = getNewEdge(); // Since twin does not exist in interior, new halfedge must be created on the exterior

	vertex* origin = twinless_edges[i].edge -> destination();
	reverse_edge -> setOrigin(origin); // Set origin of exterior edge
	reverse_edge -> setIncidentFace(exterior);
	reverse_edge -> setTwin(twinless_edges[i].edge);

	exterior_edges[origin -> index] = reverse_edge;
    }

    halfedge* firstEdge = exterior -> startEdge;
    halfedge* currEdge = firstEdge;
    do
    {
	int endIndex = currEdge -> destination() -> index;
	currEdge -> setNext(exterior_edges[endIndex]);
	currEdge = exterior_edges[endIndex];
    }
    while (currEdge != firstEdge);

    samplefaces.resize(MAX_SAMPLE_SIZE);
    int step = pow(faces.size(), 1.0 / 3.0);
    numSample = faces.size() / step;
    assert(numSample < MAX_SAMPLE_SIZE);

    #pragma omp parallel
    {
	#pragma omp for // nowait
	for (int i = 0; i < faces.size() - 1; i++) // no need to calculate the exterior's centroid, so exclude the last face
	{
	    faces[i] -> initCentroid();
	}

	#pragma omp for
	for (int i = 0; i < numSample; i++)
	{
	    samplefaces[i] = faces[i * step];
	}
    }
}

