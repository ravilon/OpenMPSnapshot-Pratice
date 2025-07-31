#pragma once

#include "InputFile.hpp"
#include "Mesh.hpp"
#include "Scheme.hpp"

// ReSharper disable once CppUnusedIncludeDirective
#include <memory>
#include <vector>

class Diffusion final {
    const std::shared_ptr<Mesh> mesh{};

    std::unique_ptr<Scheme> scheme{};

    std::vector<double> subregion{};

public:
    Diffusion() = delete;

    Diffusion(const Diffusion& other) = delete;

    Diffusion(Diffusion&& other) = delete;

    Diffusion& operator=(const Diffusion& other) const = delete;

    Diffusion(const InputFile& input, const std::shared_ptr<Mesh>& m);

    ~Diffusion() = default;

    void init() const;

    void doCycle(const double& dt) const;
};
