#pragma once

#include "Mesh.hpp"

// ReSharper disable once CppUnusedIncludeDirective
#include <memory>
#include <string>

class VtkWriter final {
    const std::string dump_basename;

    const std::shared_ptr<Mesh> mesh;

    inline static const std::string vtk_header = "# vtk DataFile Version 3.0\nvtk output\nASCII";
    inline static const std::string _path_prefix = "out/";

public:
    VtkWriter(std::string basename, const std::shared_ptr<Mesh>& mesh);

    void writeVtk(const int& step, const double& time) const;

    void writeVisited(const int& final_step) const;
};
