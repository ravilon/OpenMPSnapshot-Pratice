#pragma once

#include "vector2d.hpp"
#include "unit_system.hpp"

#include <memory>

class Particle {
    public:
        const double mass;
        Vector2D position;
        Vector2D velocity;

        Particle(const double mass, const Vector2D& position, const Vector2D& velocity,
                 const std::shared_ptr<const UnitSystem> unitSystem);

        double getKineticEnergy() const;
        double getPotentialEnergy(const Particle& particle) const;
        Vector2D getGravityAccelerationFactor(const Particle& particle) const;
        void updatePosition(const Vector2D& velocity, const double timeStep);
        void updateVelocity(const Vector2D& acceleration, const double timeStep);

    private:
        const std::shared_ptr<const UnitSystem> unitSystem;
};
