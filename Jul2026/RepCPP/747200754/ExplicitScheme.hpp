#pragma once

#include "Mesh.hpp"
#include "Scheme.hpp"

// ReSharper disable once CppUnusedIncludeDirective
#include <memory>

class ExplicitScheme final : public Scheme {
    const std::shared_ptr<Mesh> mesh{};

    void updateBoundaries(
        const int& x_min,
        const int& x_max,
        const int& y_min,
        const int& y_max,
        const int& nx
    ) const;

    void reset(
        const int& x_min,
        const int& x_max,
        const int& y_min,
        const int& y_max,
        const int& nx
    ) const;

    void diffuse(
        const double& dt,
        const int& x_min,
        const int& x_max,
        const int& y_min,
        const int& y_max,
        const int& nx
    ) const;

public:
    ExplicitScheme() = delete;

    ExplicitScheme(const ExplicitScheme& other) = delete;

    ExplicitScheme(ExplicitScheme&& other) = delete;

    ExplicitScheme& operator=(const ExplicitScheme& other) const = delete;

    explicit ExplicitScheme(const std::shared_ptr<Mesh>& m);

    ~ExplicitScheme() override = default;

    void doAdvance(const double& dt) const override;

    void init() const override;
};
