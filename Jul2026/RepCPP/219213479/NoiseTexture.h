/**
 *    Copyright 2020 Jannik Bamberger
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <array>
#include <glm/glm.hpp>
#include <random>

inline glm::dvec3 randomGradient()
{
    static std::default_random_engine rng;
    static std::uniform_real_distribution<double> d(-1.0, 1.0);

    glm::dvec3 gradient;
    do {
        gradient = glm::dvec3(d(rng), d(rng), d(rng));
    } while (glm::length(gradient) <= 1);
    return glm::normalize(gradient);
}

template <size_t ResX, size_t ResY, size_t ResZ>
class PerlinNoise {
    std::array<std::array<std::array<glm::dvec3, ResZ>, ResY>, ResX> gradients_;

  public:
    explicit PerlinNoise()
    {
        for (size_t i = 0; i < ResX; ++i) {
            for (size_t j = 0; j < ResY; ++j) {
                for (size_t k = 0; k < ResZ; ++k) {
                    gradients_[i][j][k] = randomGradient();
                }
            }
        }
    }

  private:
    constexpr static double lerp(const double a0, const double a1, const double w)
    {
        return a0 + w * (a1 - a0);
    }

    double grad(const glm::dvec3 pos, int x, int y, int z) const
    {
        const auto dist = glm::dvec3(pos.x - static_cast<double>(x), pos.y - static_cast<double>(y),
                                     pos.z - static_cast<double>(z));
        return glm::dot(dist, gradients_[x][y][z]);
    }

  public:
    double value(const glm::dvec3 pos) const
    {
        // lower cell bound
        const auto x0 = static_cast<int>(pos.x);
        const auto y0 = static_cast<int>(pos.y);
        const auto z0 = static_cast<int>(pos.z);
        // upper cell bound
        const auto x1 = x0 + 1;
        const auto y1 = y0 + 1;
        const auto z1 = z0 + 1;

        // interpolation weights
        const auto wx = pos.x - static_cast<double>(x0);
        const auto wy = pos.y - static_cast<double>(y0);
        const auto wz = pos.z - static_cast<double>(z0);

        // dot products
        const auto p0 = grad(pos, x0, y0, z0);
        const auto p1 = grad(pos, x1, y0, z0);
        const auto p2 = grad(pos, x0, y1, z0);
        const auto p3 = grad(pos, x1, y1, z0);
        const auto p4 = grad(pos, x0, y0, z1);
        const auto p5 = grad(pos, x1, y0, z1);
        const auto p6 = grad(pos, x0, y1, z1);
        const auto p7 = grad(pos, x1, y1, z1);

        // interpolation in x direction
        const auto q0 = lerp(p0, p1, wx);
        const auto q1 = lerp(p2, p3, wx);
        const auto q2 = lerp(p4, p5, wx);
        const auto q3 = lerp(p6, p7, wx);

        // interpolation in y direction
        const auto r0 = lerp(q0, q1, wy);
        const auto r1 = lerp(q2, q3, wy);

        // interpolation in z direction
        return lerp(r0, r1, wz);
    }
};

template <size_t ResX, size_t ResY, size_t ResZ>
class NoiseTexture {
    PerlinNoise<ResX, ResY, ResZ> noise_;
    glm::dvec3 min_;
    glm::dvec3 rescale_;

  public:
    NoiseTexture(const glm::dvec3& min, const glm::dvec3& max)
        : min_(min), rescale_(glm::dvec3(1.0, 1.0, 1.0) /
                              ((max - min) * glm::dvec3(ResX - 1, ResY - 1, ResZ - 1)))
    {
    }

    double value(const glm::dvec3 pos) const
    {
        // map the position into the grid of the noise function:
        // new_pos = (pos - min) / (max - min) * grid_size
        const auto v = (1 + noise_.value((pos - min_) * rescale_)) / 2;
        assert(0.0 <= v && v <= 1.0);
        return glm::clamp(v, 0.0, 1.0);
    }
};
