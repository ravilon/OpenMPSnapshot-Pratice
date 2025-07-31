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
#include <memory>
#include <ostream>
#include <vector>

/**
 * This class is a simple compound entity which does not use any form of hierarchic optimization.
 * The elements are stored in a vector and checked all at once.
 */
struct ExplicitEntity final : Entity {
    explicit ExplicitEntity(std::vector<Triangle> faces);

    void setMaterial(std::shared_ptr<Material> material) override;

    [[nodiscard]] bool intersect(const Ray& ray, Hit& hit) const override;

    [[nodiscard]] BoundingBox boundingBox() const override;

  private:
    std::vector<Triangle> faces_;
    BoundingBox bbox_;
};
