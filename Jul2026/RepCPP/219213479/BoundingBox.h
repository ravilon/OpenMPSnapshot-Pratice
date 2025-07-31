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

#include "Ray.h"
#include <glm/glm.hpp>

/**
 * This class represents an axis-aligned bounding box.
 */
struct BoundingBox {

    /**
     * Lower left back corner.
     */
    glm::dvec3 min;

    /**
     * Upper right front corner.
     */
    glm::dvec3 max;

    /**
     * Constructs a new axis-aligned bounding box with the given corner points.
     * @param min min coordinates in every direction
     * @param max max coordinate in every direction
     */
    BoundingBox(glm::dvec3 min, glm::dvec3 max);

    /**
     * Checks if this bbox and another bounding box intersect.
     * @param other the other bbox
     * @return true if the boxes intersect
     */
    [[nodiscard]] bool intersect(const BoundingBox& other) const;

    /**
     * Checks if the ray intersects this box in any location.
     * @param ray the ray
     * @return true if the ray intersects the bbox
     */
    [[nodiscard]] bool intersect(const Ray& ray) const;

    /**
     * Check if the point lies within this bounding box.
     * @param point the point
     * @return true if the point is inside of the bbox
     */
    [[nodiscard]] bool contains(glm::dvec3 point) const;

    /**
     * Computes the tightest fitting bbox which encloses both b1 and b2.
     * @param b1 first bbox
     * @param b2 second bbox
     * @return box containing b1 and b2
     */
    static BoundingBox unite(const BoundingBox& b1, const BoundingBox& b2);
};
