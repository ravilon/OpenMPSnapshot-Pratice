/**
 * @file    tree_mesh_builder.cpp
 *
 * @author  Ondrej Vlcek <xvlcek27@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 **/

#include <iostream>
#include <math.h>
#include <limits>
#include <omp.h>

#include "tree_mesh_builder.h"

TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
    : mTriangles(omp_get_max_threads()), BaseMeshBuilder(gridEdgeSize, "Octree")
{
    //mTriangles = std::vector<std::vector<BaseMeshBuilder::Triangle_t>>(omp_get_max_threads());
}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    unsigned ret = 0;
    #pragma omp parallel
    {
        #pragma omp master
        {
            ret = OctreeDescent(field, 0, mGridSize, 0, 0, 0);
        }
    }
    return ret;
}

float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{
    const Vec3_t<float> *pPoints = field.getPoints().data();
    const unsigned count = unsigned(field.getPoints().size());

    float value = std::numeric_limits<float>::max();

    for(unsigned i = 0; i < count; ++i)
    {
        float distanceSquared  = (pos.x - pPoints[i].x) * (pos.x - pPoints[i].x);
        distanceSquared       += (pos.y - pPoints[i].y) * (pos.y - pPoints[i].y);
        distanceSquared       += (pos.z - pPoints[i].z) * (pos.z - pPoints[i].z);

        value = std::min(value, distanceSquared);
    }

    return sqrt(value);
}

void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
{
    mTriangles[omp_get_thread_num()].push_back(triangle);
}

const float sqrt3 = sqrt(3.f);
const int offsets[] = {
    0,0,0,
    1,0,0,
    0,1,0,
    1,1,0,
    0,0,1,
    1,0,1,
    0,1,1,
    1,1,1
};

unsigned TreeMeshBuilder::OctreeDescent(const ParametricScalarField &field, int depth, int sidelen, int x, int y, int z) {
    if (sidelen == 1) {
        return buildCube(Vec3_t<float>(x, y, z), field);
    }

    int newSidelen = sidelen / 2;
    float eval = evaluateFieldAt(Vec3_t<float>(
        (x + newSidelen) * mGridResolution,
        (y + newSidelen) * mGridResolution,
        (z + newSidelen) * mGridResolution
    ), field);

    if (eval > mIsoLevel + (sqrt3 * mGridResolution * newSidelen)) {
        return 0;
    }

    unsigned ret = 0;

    for (int i = 0; i < 8; i++) {
        #pragma omp task shared(ret) final(depth >= 4)
        {
            #pragma omp atomic update
            ret += OctreeDescent(field, depth+1, newSidelen,
                x + offsets[3*i] * newSidelen,
                y + offsets[3*i+1] * newSidelen,
                z + offsets[3*i+2] * newSidelen
            );
        }
    }

    #pragma omp taskwait

    return ret;
}
