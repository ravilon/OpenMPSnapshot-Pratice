#include "VtkWriter.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

VtkWriter::VtkWriter(std::string basename, const std::shared_ptr<Mesh>& mesh)
    : dump_basename{std::move(basename)},
      mesh{mesh} {
#ifdef DEBUG
    if (!std::filesystem::exists(_path_prefix) || !std::filesystem::is_directory(_path_prefix)) {
        std::filesystem::create_directory(_path_prefix);
    }
#endif
}

void VtkWriter::writeVtk(const int& step, const double& time) const {
    const int x_coordinate = mesh->getNx()[0] + 1;
    const int y_coordinate = mesh->getNx()[1] + 1;
    const int cell_data_x = x_coordinate - 1;
    const int cell_data_y = y_coordinate - 1;
    const int total_cells = cell_data_x * cell_data_y;

    std::vector<std::string> x_coord_texts{static_cast<std::vector<std::string>::size_type>(x_coordinate)};
    std::vector<std::string> y_coord_texts{static_cast<std::vector<std::string>::size_type>(y_coordinate)};
    std::vector<std::string> u_data_texts{static_cast<std::vector<std::string>::size_type>(cell_data_y)};

    // ReSharper disable CppDFAUnusedValue CppDFAUnreadVariable CppDFALoopConditionNotUpdated
#pragma omp parallel default(none) shared(mesh, x_coordinate, y_coordinate, cell_data_x, cell_data_y, x_coord_texts, y_coord_texts, u_data_texts)
    {
#pragma omp for nowait schedule(static)
        for (int i = 0; i < x_coordinate; ++i) {
            std::ostringstream oss;
            oss.precision(8);
            oss.setf(std::ios::fixed, std::ios::floatfield);
            oss << mesh->getCellX()[i + 1] << " ";
            x_coord_texts[i] = oss.view();
        }

#pragma omp for nowait schedule(static)
        for (int i = 0; i < y_coordinate; ++i) {
            std::ostringstream oss;
            oss.precision(8);
            oss.setf(std::ios::fixed, std::ios::floatfield);
            oss << mesh->getCellY()[i + 1] << " ";
            y_coord_texts[i] = oss.view();
        }

#pragma omp for nowait schedule(static)
        for (int j = 1; j <= cell_data_y; j++) {
            std::ostringstream oss;
            oss.precision(8);
            oss.setf(std::ios::fixed, std::ios::floatfield);

            for (int i = 1; i <= cell_data_x; i++) {
                oss << mesh->getU0()[i + j * (cell_data_x + 2)] << " ";
            }

            oss << '\n';
            u_data_texts[j - 1] = oss.view();
        }
    }

    std::stringstream file_buffer;

    file_buffer
            << vtk_header << '\n'
            << "DATASET RECTILINEAR_GRID" << '\n'
            << "FIELD FieldData 2" << '\n'
            << "TIME 1 1 double" << '\n'
            << time << '\n'
            << "CYCLE 1 1 int" << '\n'
            << step << '\n'
            << "DIMENSIONS " << x_coordinate << " " << y_coordinate << " 1" << '\n'
            << "X_COORDINATES " << x_coordinate << " float" << '\n';

    for (const auto& x: x_coord_texts) file_buffer << x;

    file_buffer << "Y_COORDINATES " << y_coordinate << " float" << '\n';

    for (const auto& y: y_coord_texts) file_buffer << y;

    file_buffer
            << "Z_COORDINATES 1 float" << '\n'
            << "0.0000" << '\n'
            << "CELL_DATA " << cell_data_x * cell_data_y << '\n'
            << "FIELD FieldData 1" << '\n'
            << "u 1 " << total_cells << " double" << '\n';

    for (const auto& u: u_data_texts) file_buffer << u;

    std::stringstream fname;

    fname
            << _path_prefix
            << dump_basename
            << "."
            << step
            << "."
            << 1
            << ".vtk";

    std::ios::sync_with_stdio(false);

    std::ofstream file;
    file.open(fname.str(), std::ofstream::binary | std::ofstream::out | std::ofstream::trunc);
    file.setf(std::ios::fixed, std::ios::floatfield);
    file.precision(8);

    if (file.is_open()) {
        file << file_buffer.rdbuf();
        file.close();
    }
}

void VtkWriter::writeVisited(const int& final_step) const {
    std::stringstream visited_names;
    for (int step = 1; step <= final_step; ++step) {
        visited_names
                << dump_basename
                << "."
                << step
                << "."
                << 1
                << ".vtk" << '\n';
    }

    std::ios::sync_with_stdio(false);
    std::ofstream file;
    file.open(
        _path_prefix + dump_basename + ".visit",
        std::ofstream::binary | std::ofstream::out | std::ofstream::trunc
    );

    if (file.is_open()) {
        file
                << "!NBLOCKS " << 1 << '\n'
                << visited_names.rdbuf();
        file.close();
    }
}
