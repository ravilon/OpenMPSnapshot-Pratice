/**
* @file    tree_mesh_builder.cpp
*
* @author  Dominik Harmim <xharmi00@stud.fit.vutbr.cz>
*
* @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree
*          early elimination.
*
* @date    12 December 2019, 17:17
**/


#include <cmath>
#include <limits>

#include "tree_mesh_builder.h"


TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize) :
BaseMeshBuilder(gridEdgeSize, "Octree")
{
}


auto TreeMeshBuilder::marchCubes(const ParametricScalarField &field) -> unsigned
{
unsigned trianglesCount = 0;

#pragma omp parallel default(none) shared(trianglesCount, field)
#pragma omp single nowait
trianglesCount = decomposeSpace(mGridSize, Vec3_t<float>(), field);

return trianglesCount;
}


auto TreeMeshBuilder::decomposeSpace(
const unsigned gridSize,
const Vec3_t<float> &cubeOffset,
const ParametricScalarField &field
) -> unsigned
{
const auto edgeLength = float(gridSize);
if (isBlockEmpty(edgeLength, cubeOffset, field))
{
return 0;
}

if (gridSize <= GRID_SIZE_CUT_OFF)
{
return buildCube(cubeOffset, field);
}

unsigned totalTrianglesCount = 0;
const unsigned newGridSize = gridSize / 2;
const auto newEdgeLength = float(newGridSize);

for (const Vec3_t<float> vertexNormPos : sc_vertexNormPos)
{
#pragma omp task default(none) firstprivate(vertexNormPos)  shared(cubeOffset, newEdgeLength, newGridSize, field, totalTrianglesCount)
{
const Vec3_t<float> newCubeOffset(
cubeOffset.x + vertexNormPos.x * newEdgeLength,
cubeOffset.y + vertexNormPos.y * newEdgeLength,
cubeOffset.z + vertexNormPos.z * newEdgeLength
);
const unsigned trianglesCount =
decomposeSpace(newGridSize, newCubeOffset, field);

#pragma omp atomic update
totalTrianglesCount += trianglesCount;
}
}

#pragma omp taskwait
return totalTrianglesCount;
}


auto TreeMeshBuilder::isBlockEmpty(
const float edgeLength,
const Vec3_t<float> &cubeOffset,
const ParametricScalarField &field
) -> bool
{
const float resEdgeLength = edgeLength * mGridResolution;
const float halfEdgeLength = resEdgeLength / 2.F;
const Vec3_t<float> midPoint(
cubeOffset.x * mGridResolution + halfEdgeLength,
cubeOffset.y * mGridResolution + halfEdgeLength,
cubeOffset.z * mGridResolution + halfEdgeLength
);
static const float expr = sqrtf(3.F) / 2.F;

return evaluateFieldAt(midPoint, field) > mIsoLevel + expr * resEdgeLength;
}


auto TreeMeshBuilder::evaluateFieldAt(
const Vec3_t<float> &pos, const ParametricScalarField &field
) -> float
{
float minDistanceSquared = std::numeric_limits<float>::max();

for (const Vec3_t<float> point : field.getPoints())
{
const float distanceSquared = (pos.x - point.x) * (pos.x - point.x)
+ (pos.y - point.y) * (pos.y - point.y)
+ (pos.z - point.z) * (pos.z - point.z);
minDistanceSquared = std::min(minDistanceSquared, distanceSquared);
}

return sqrtf(minDistanceSquared);
}


void TreeMeshBuilder::emitTriangle(const Triangle_t &triangle)
{
#pragma omp critical(tree_emitTriangle)
triangles.push_back(triangle);
}
