// TODO: remove the following rule disable and fix the issue
// ReSharper disable CppDFAUnusedValue CppDFAUnreadVariable CppDFALoopConditionNotUpdated
#include "ExplicitScheme.hpp"

#include <iostream>

ExplicitScheme::ExplicitScheme(const std::shared_ptr<Mesh>& m) : mesh(m) {
}

void ExplicitScheme::init() const {
    const int x_min = mesh->getMin()[0];
    const int x_max = mesh->getMax()[0];
    const int y_min = mesh->getMin()[1];
    const int y_max = mesh->getMax()[1];
    const int nx = mesh->getNx()[0] + 2;

    updateBoundaries(x_min, x_max, y_min, y_max, nx);
}

void ExplicitScheme::doAdvance(const double& dt) const {
    const int x_min = mesh->getMin()[0];
    const int x_max = mesh->getMax()[0];
    const int y_min = mesh->getMin()[1];
    const int y_max = mesh->getMax()[1];
    const int nx = mesh->getNx()[0] + 2;

    diffuse(dt, x_min, x_max, y_min, y_max, nx);
    reset(x_min, x_max, y_min, y_max, nx);
    updateBoundaries(x_min, x_max, y_min, y_max, nx);
}

void ExplicitScheme::updateBoundaries(
    const int& x_min,
    const int& x_max,
    const int& y_min,
    const int& y_max,
    const int& nx
) const {
    auto& u0 = mesh->getU0();

#pragma omp parallel default(none) shared(u0) firstprivate(x_min, x_max, y_min, y_max, nx)
    {
#pragma omp for schedule(static) nowait // Edge -> top
        for (int i = x_min; i <= x_max; i++) {
            const int n1 = Mesh::poly2(i, y_max, x_min - 1, y_min - 1, nx);
            const int n2 = Mesh::poly2(i, y_max + 1, x_min - 1, y_min - 1, nx);

            u0[n2] = u0[n1];
        }

#pragma omp for schedule(static) nowait // Edge -> right
        for (int j = y_min; j <= y_max; j++) {
            const int n1 = Mesh::poly2(x_max, j, x_min - 1, y_min - 1, nx);
            const int n2 = Mesh::poly2(x_max + 1, j, x_min - 1, y_min - 1, nx);

            u0[n2] = u0[n1];
        }

#pragma omp for schedule(static) nowait // Edge -> bottom
        for (int i = x_min; i <= x_max; i++) {
            const int n1 = Mesh::poly2(i, y_min, x_min - 1, y_min - 1, nx);
            const int n2 = Mesh::poly2(i, y_min - 1, x_min - 1, y_min - 1, nx);

            u0[n2] = u0[n1];
        }

#pragma omp for schedule(static) nowait // Edge -> left
        for (int j = y_min; j <= y_max; j++) {
            const int n1 = Mesh::poly2(x_min, j, x_min - 1, y_min - 1, nx);
            const int n2 = Mesh::poly2(x_min - 1, j, x_min - 1, y_min - 1, nx);

            u0[n2] = u0[n1];
        }
    }
}

void ExplicitScheme::reset(
    const int& x_min,
    const int& x_max,
    const int& y_min,
    const int& y_max,
    const int& nx
) const {
    auto& u0 = mesh->getU0();
    const auto& u1 = mesh->getU1();

#pragma omp parallel default(none) shared(u0, u1) firstprivate(x_min, x_max, y_min, y_max, nx)
    {
#pragma omp for collapse(2) schedule(static) nowait
        for (int k = y_min - 1; k <= y_max + 1; k++) {
            for (int j = x_min - 1; j <= x_max + 1; j++) {
                const int i = Mesh::poly2(j, k, x_min - 1, y_min - 1, nx);
                u0[i] = u1[i];
            }
        }
    }
}

void ExplicitScheme::diffuse(
    const double& dt,
    const int& x_min,
    const int& x_max,
    const int& y_min,
    const int& y_max,
    const int& nx
) const {
    const auto& u0 = mesh->getU0();
    auto& u1 = mesh->getU1();

    const double dx = mesh->getDx()[0];
    const double dy = mesh->getDx()[1];

    const double rx = dt / (dx * dx);
    const double ry = dt / (dy * dy);

#pragma omp parallel default(none) shared(u0, u1) firstprivate(x_min, x_max, y_min, y_max, nx, rx, ry)
    {
#pragma omp for collapse(2) schedule(static) nowait
        for (int k = y_min; k <= y_max; k++) {
            for (int j = x_min; j <= x_max; j++) {
                const int n1 = Mesh::poly2(j, k, x_min - 1, y_min - 1, nx);
                const int n2 = Mesh::poly2(j - 1, k, x_min - 1, y_min - 1, nx);
                const int n3 = Mesh::poly2(j + 1, k, x_min - 1, y_min - 1, nx);
                const int n4 = Mesh::poly2(j, k - 1, x_min - 1, y_min - 1, nx);
                const int n5 = Mesh::poly2(j, k + 1, x_min - 1, y_min - 1, nx);

                u1[n1] = (1.0 - 2.0 * rx - 2.0 * ry) * u0[n1] +
                         rx * u0[n2] +
                         rx * u0[n3] +
                         ry * u0[n4] +
                         ry * u0[n5];
            }
        }
    }
}
