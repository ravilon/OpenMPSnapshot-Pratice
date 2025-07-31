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

#include "Entity.h"
#include "ExplicitEntity.h"
#include <array>

namespace entities {
/// Creates an explicit sphere from a center and radius by iteratively subdividing each triangle
/// and projecting it to the implicit sphere. The starting shape is an icosahedron or a
/// tetrahedron.
std::unique_ptr<ExplicitEntity> makeSphere(glm::dvec3 center = {0, 0, 0},
                                           double radius = 1,
                                           int sub_divisions = 2,
                                           bool use_tetrahedron = false);

/// Creates an explicit quad from the four points.
std::unique_ptr<ExplicitEntity> makeQuad(glm::dvec3 a, glm::dvec3 b, glm::dvec3 c, glm::dvec3 d);

/// Creates an explicit axis-aligned cube.
std::unique_ptr<ExplicitEntity> makeCube(glm::dvec3 center, double side_length);

/// Creates an explict cuboid
std::unique_ptr<ExplicitEntity> makeCuboid(glm::dvec3 center, glm::dvec3 size);

/// Creates an explict cuboid
std::unique_ptr<ExplicitEntity> makeOctet(std::array<glm::dvec3, 8> corners);

/// Creates an explicit cone.
std::unique_ptr<ExplicitEntity> makeCone(glm::dvec3 center,
                                         glm::dvec3 tip,
                                         double radius,
                                         size_t slices);

} // namespace entities
