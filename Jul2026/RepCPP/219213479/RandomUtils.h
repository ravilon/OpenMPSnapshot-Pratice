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

#include <chrono>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <random>

/**
 * Returns a random number between 0 and 1. The number is generated from a thread-local rng.
 * @return random number in the interval [0,1]
 */
inline double rng()
{
    static thread_local std::default_random_engine engine(
        static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()));

    std::uniform_real_distribution<double> dist(0, 1);
    return dist(engine);
}

/**
 * Computes a random point in the unit sphere. The point is computed directly.
 *
 * @return random point in the unit sphere.
 */
inline glm::dvec3 randomOffset()
{
    //    glm::dvec3 p;
    //    do {
    //        p = glm::dvec3(rng(), rng(), rng()) * 2.0 - glm::dvec3(1, 1, 1);
    //    } while (glm::dot(p, p) >= 1.0);
    //    return p;

    glm::dvec3 vec;
    auto r1 = rng();
    auto r2 = rng();
    vec.x = glm::cos(glm::two_pi<double>() * r1) * 2 * glm::sqrt(r2 * (1 - r2));
    vec.y = glm::sin(glm::two_pi<double>() * r1) * 2 * glm::sqrt(r2 * (1 - r2));
    vec.z = 1 - 2 * r2;
    return vec;
}

/**
 * Compute a random point in the hemisphere given by the normal vector and the incoming ray
 * direction.
 *
 * @param normal surface normal
 * @param direction ray direction
 * @return random point in the hemisphere
 */
inline glm::dvec3 hemisphere(const glm::dvec3 normal, const glm::dvec3 direction)
{

    const auto r1 = glm::two_pi<double>() * rng();
    const auto r2 = rng();
    const auto sq2 = glm::sqrt(r2);

    const auto x = glm::cos(r1) * sq2;
    const auto y = glm::sin(r1) * sq2;
    const auto z = glm::sqrt(1 - r2);

    // build basis vectors around normal vector
    const auto w = glm::dot(normal, direction) < 0 ? normal : -normal;
    glm::dvec3 c;
    if (std::abs(w.x) < 1 / glm::sqrt(3)) {
        c = glm::dvec3(1, 0, 0);
    } else if (std::abs(w.y) < 1 / glm::sqrt(3)) {
        c = glm::dvec3(0, 1, 0);
    } else {
        c = glm::dvec3(0, 0, 1);
    }
    const auto u = glm::cross(c, w);
    const auto v = glm::cross(w, u);

    // shirley 14. Sampling p294 short form which avoid duplicate use of trig function
    return glm::normalize(u * x + v * y + w * z);
}
