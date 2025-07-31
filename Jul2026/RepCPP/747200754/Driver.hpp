#pragma once

#include "Diffusion.hpp"
#include "InputFile.hpp"
#include "Mesh.hpp"
#include "VtkWriter.hpp"

// ReSharper disable once CppUnusedIncludeDirective
#include <memory>
#include <string>

class Driver final {
    const std::shared_ptr<Mesh> mesh{};

    const Diffusion diffusion;

    const VtkWriter writer;

    const std::string _problem_name;

    double t_start, t_end, dt_max, dt;

    int vis_frequency, summary_frequency;

public:
    Driver() = delete;

    Driver(const Driver& other) = delete;

    Driver(Driver&& other) = delete;

    Driver& operator=(const Driver& other) const = delete;

    Driver(const InputFile& input, const std::string& problem_name);

    ~Driver() = default;

    void run() const;
};
