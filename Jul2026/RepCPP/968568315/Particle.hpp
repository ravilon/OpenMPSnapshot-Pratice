#pragma once
#include <cstdint>

/**
 * @brief Core particle data structure
 * 
 * Compact particle representation optimized for SIMD operations
 * and cache efficiency. Total size: 4 bytes per particle.
 * 
 * Memory layout:
 * - type: 1 byte (particle type enum)
 * - mass: 1 byte (0-255 range)
 * - velocity: 2 bytes (x,y components)
 * 
 * Usage:
 * @code
 * Particle sand(ParticleType::SAND, 100);
 * if(particle.isEmpty()) {
 *     // Handle empty cell
 * }
 * @endcode
 */
enum class ParticleType : uint8_t {
    EMPTY = 0,
    SAND,
    WATER,
    STONE,
    WOOD
};

struct Particle {
    ParticleType type = ParticleType::EMPTY;
    uint8_t mass = 0;
    uint8_t velocity_x = 0;
    uint8_t velocity_y = 0;
    
    // Default constructor creates an empty particle
    Particle() = default;
    
    /**
     * @brief Creates particle of specific type
     * @param t Particle type
     * @param m Mass value (default 1)
     */
    explicit Particle(ParticleType t, uint8_t m = 1)
        : type(t)
        , mass(m)
        , velocity_x(0)
        , velocity_y(0)
    {}
   
    bool isEmpty() const {
        return type == ParticleType::EMPTY;
    }
};
