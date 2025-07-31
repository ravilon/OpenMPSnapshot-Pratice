/**
 * @file    loop_mesh_builder.cpp
 *
 * @author  ONDŘEJ KREJČÍ <xkrejc69@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP loops
 *
 * @date    17. 12. 2021
 **/

#include <iostream>
#include <math.h>
#include <limits>
#include <omp.h>

#include "loop_mesh_builder.h"

LoopMeshBuilder::LoopMeshBuilder(unsigned gridEdgeSize)
    : BaseMeshBuilder(gridEdgeSize, "OpenMP Loop")
{
    mTriangleVectors = new std::vector<Triangle_t>[threads];
}

LoopMeshBuilder::~LoopMeshBuilder(){
    delete[] mTriangleVectors;

    if(x != nullptr){
        _mm_free(x);
        _mm_free(y);
        _mm_free(z);
    }
}

unsigned LoopMeshBuilder::marchCubes(const ParametricScalarField& field)
{
    fieldToArrays(field);

    // Loop over each coordinate in the 3D grid.
    #pragma omp parallel default(none) shared(field)
    {
        #pragma omp for schedule(static) collapse(3)
        for(size_t x = 0; x < mGridSize; x++){
            for(size_t y = 0; y < mGridSize; y++){
                for(size_t z = 0; z < mGridSize; z++){
                    buildCube(Vec3_t<float>{x, y, z}, field);
                }
            }
        }
    }

    for(unsigned i = 0; i < threads; i++){
        mTriangles.insert(mTriangles.end(), mTriangleVectors[i].begin(), mTriangleVectors[i].end());
    }

    // Return total number of triangles generated.
    return mTriangles.size();
}

// NOTE: This method is called from "buildCube(...)"!
float LoopMeshBuilder::evaluateFieldAt(const Vec3_t<float>& pos, const ParametricScalarField& field)
{
    float value = std::numeric_limits<float>::max();

    const float* lx = x;
    const float* ly = y;
    const float* lz = z;

    // Find minimum square distance from points "pos" to any point in the field.
    //#pragma omp parallel for default(none) shared(value, pPoints, pos) reduction(min:value) schedule(static)
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
void LoopMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t& triangle)
{
    // Store generated triangle
    mTriangleVectors[omp_get_thread_num()].push_back(triangle);
}

inline void LoopMeshBuilder::fieldToArrays(const ParametricScalarField& field){
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
