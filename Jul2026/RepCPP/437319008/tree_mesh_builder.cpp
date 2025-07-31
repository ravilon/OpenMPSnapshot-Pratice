/**
 * @file    tree_mesh_builder.cpp
 *
 * @author  ONDŘEJ KREJČÍ <xkrejc69@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    17. 12. 2021
 **/

#include <iostream>
#include <math.h>
#include <limits>

#include "tree_mesh_builder.h"

TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
    : BaseMeshBuilder(gridEdgeSize, "Octree")
{
    mTriangleVectors = new std::vector<Triangle_t>[threads];
}

TreeMeshBuilder::~TreeMeshBuilder(){
    delete[] mTriangleVectors;

    if(x != nullptr){
        _mm_free(x);
        _mm_free(y);
        _mm_free(z);
    }
}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField& field)
{
    // Suggested approach to tackle this problem is to add new method to
    // this class. This method will call itself to process the children.
    // It is also strongly suggested to first implement Octree as sequential
    // code and only when that works add OpenMP tasks to achieve parallelism.

    fieldToArrays(field);

    #pragma omp parallel default(none) shared(field)
    {
        #pragma omp single
        {
            octree(field, Vec3_t<float>{0, 0, 0}, mGridSize);
        }
    }

    for(unsigned i = 0; i < threads; i++){
        mTriangles.insert(mTriangles.end(), mTriangleVectors[i].begin(), mTriangleVectors[i].end());
    }

    // Return total number of triangles generated.
    return mTriangles.size();
}

// NOTE: This method is called from "buildCube(...)"!
float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float>& pos, const ParametricScalarField& field)
{
    float value = std::numeric_limits<float>::max();

    const float* lx = x;
    const float* ly = y;
    const float* lz = z;

    // Find minimum square distance from points "pos" to any point in the field.
    #pragma omp simd reduction(min:value) aligned(lx:64, ly:64, lz:64)
    for(unsigned i = 0; i < fieldSize; i++)
    {
        float distanceSquared  = (pos.x - x[i]) * (pos.x - x[i]);
        distanceSquared       += (pos.y - y[i]) * (pos.y - y[i]);
        distanceSquared       += (pos.z - z[i]) * (pos.z - z[i]);

        // Comparing squares instead of real distance to avoid unnecessary "sqrt"s in the loop.
        value = std::min(value, distanceSquared);
    }

    // Finally take square root of the minimal square distance to get the real distance
    return sqrt(value);
}

// NOTE: This method is called from "buildCube(...)"!
void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t& triangle)
{
    // Store generated triangle
    mTriangleVectors[omp_get_thread_num()].push_back(triangle);
}

void TreeMeshBuilder::octree(const ParametricScalarField &field, const Vec3_t<float>& start, unsigned len){
    if(len == 1){
        buildCube(start, field);
        return;
    }

    const unsigned half = len >> 1;

    const std::vector<Vec3_t<float>> positions{
        {start.x, start.y, start.z},
        {start.x + half, start.y, start.z},
        {start.x, start.y + half, start.z},
        {start.x + half, start.y + half, start.z},
        {start.x, start.y, start.z + half},
        {start.x + half, start.y, start.z + half},
        {start.x, start.y + half, start.z + half},
        {start.x + half, start.y + half, start.z + half}
    };

    for(const auto& position : positions){
        if(isEmpty(field, position, half)){
            return;
        }

        #pragma omp task default(none) shared(field, half) firstprivate(position) if(half > 1)
        {
            octree(field, position, half);
        }
    }

    #pragma omp taskwait
    return;
}

bool TreeMeshBuilder::isEmpty(const ParametricScalarField& field, const Vec3_t<float>& start, unsigned len){
    const float y = mIsoLevel + (HALF_SQRT3 * len);
    const unsigned half = len >> 1;
    const float fp = evaluateFieldAt(Vec3_t<float>{(start.x + half) * mGridResolution, (start.y + half) * mGridResolution, (start.z + half) * mGridResolution}, field);

    return (fp > y);
}

inline void TreeMeshBuilder::fieldToArrays(const ParametricScalarField& field){
    const Vec3_t<float>* pPoints = field.getPoints().data();

    fieldSize = unsigned(field.getPoints().size());

    x = (float*) _mm_malloc(fieldSize * sizeof(float), 64);
    y = (float*) _mm_malloc(fieldSize * sizeof(float), 64);
    z = (float*) _mm_malloc(fieldSize * sizeof(float), 64);

    #pragma omp parallel for default(none) shared(fieldSize, pPoints, x, y, z) if(fieldSize > 10000)
    for(unsigned i = 0; i < fieldSize; i++){
        x[i] = pPoints[i].x;
        y[i] = pPoints[i].y;
        z[i] = pPoints[i].z;
    }
}
