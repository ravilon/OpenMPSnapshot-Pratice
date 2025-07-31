// TODO: remove the following rule disable and fix the issue
// ReSharper disable CppDFAUnusedValue CppDFAUnreadVariable CppDFALoopConditionNotUpdated
#include "Mesh.hpp"

Mesh::Mesh(const InputFile& inputFile) {
    const int nx = inputFile.getInt("nx", 0);
    const int ny = inputFile.getInt("ny", 0);

    min_coords[0] = inputFile.getDouble("xmin", 0.0);
    max_coords[0] = inputFile.getDouble("xmax", 1.0);
    min_coords[1] = inputFile.getDouble("ymin", 0.0);
    max_coords[1] = inputFile.getDouble("ymax", 1.0);

    // setup first dimension.
    n[0] = nx;
    min[0] = 1;
    max[0] = nx;

    dx[0] = (max_coords[0] - min_coords[0]) / nx;

    // setup second dimension.
    n[1] = ny;
    min[1] = 1;
    max[1] = ny;

    dx[1] = (max_coords[1] - min_coords[1]) / ny;

    allocate();
}

void Mesh::allocate() {
    const int nx = n[0];
    const int ny = n[1];

    u1.clear();
    u1.resize((nx + 2) * (ny + 2));

    u0.clear();
    u0.resize((nx + 2) * (ny + 2));

    cellx.clear();
    cellx.resize(nx + 2);

    celly.clear();
    celly.resize(ny + 2);

    const double xmin = min_coords[0];
    const double ymin = min_coords[1];

#pragma omp parallel default(none) shared(cellx, celly, dx, xmin, ymin)
    {
#pragma omp for schedule(static) nowait
        for (int i = 0; i < static_cast<int>(cellx.size()); ++i) {
            cellx[i] = xmin + dx[0] * (i - 1);
        }

#pragma omp for schedule(static) nowait
        for (int i = 0; i < static_cast<int>(celly.size()); ++i) {
            celly[i] = ymin + dx[1] * (i - 1);
        }
    }

    allocated = true;
}

std::vector<double>& Mesh::getU0() {
    return u0;
}

std::vector<double>& Mesh::getU1() {
    return u1;
}

const std::array<double, NDIM>& Mesh::getDx() const {
    return dx;
}

const std::array<int, NDIM>& Mesh::getMin() const {
    return min;
}

const std::array<int, NDIM>& Mesh::getMax() const {
    return max;
}

int Mesh::getDim() {
    return NDIM;
}

const std::array<int, NDIM>& Mesh::getNx() const {
    return n;
}

const std::array<int, 4>& Mesh::getNeighbours() const {
    return neighbours;
}

const std::vector<double>& Mesh::getCellX() const {
    return cellx;
}

const std::vector<double>& Mesh::getCellY() const {
    return celly;
}

double Mesh::getTotalTemperature() const {
    if (!allocated) {
        return 0.0;
    }

    double temperature = 0.0;
    const int x_min = min[0];
    const int x_max = max[0];
    const int y_min = min[1];
    const int y_max = max[1];

    const int nx = n[0] + 2;

#pragma omp parallel default(none) shared(temperature, u0, x_min, x_max, y_min, y_max, nx)
    {
#pragma omp for collapse(2) schedule(static) reduction(+ : temperature) nowait
        for (int k = y_min; k <= y_max; k++) {
            for (int j = x_min; j <= x_max; j++) {
                const int n1 = poly2(j, k, x_min - 1, y_min - 1, nx);
                temperature += u0[n1];
            }
        }
    }

    return temperature;
}
