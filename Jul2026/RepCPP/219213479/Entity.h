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

#include "BoundingBox.h"
#include "Ray.h"
#include <cassert>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/epsilon.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <iostream>
#include <memory>
#include <utility>

class Material;

struct Hit {
    glm::dvec3 normal{};
    glm::dvec3 pos{}; // hit position
    glm::dvec2 uv{};  // uv coordinates of the hit
    std::shared_ptr<Material> mat;
    Hit();
};

class Hittable {
  public:
    virtual ~Hittable() = default;
    [[nodiscard]] virtual bool intersect(const Ray& ray, Hit& hit) const = 0;
    [[nodiscard]] virtual BoundingBox boundingBox() const = 0;
};

/// A base class for all entities in the scene.
class Entity : public Hittable {
  protected:
    std::shared_ptr<Material> material_;

  public:
    explicit Entity();
    explicit Entity(std::shared_ptr<Material> material);
    ~Entity() override = default;
    virtual void setMaterial(std::shared_ptr<Material> material);
};

class Triangle final : public Entity {
    /// texture coordinates
    glm::dvec2 tA = {0, 0};
    glm::dvec2 tAB = {0, 0};
    glm::dvec2 tAC = {0, 0};

    glm::dmat2x3 to_tex_map;
    //    BoundingBox bbox_;

  public:
    /// corner points (ccw)
    glm::dvec3 A = {0, 0, 0};
    glm::dvec3 B = {0, 0, 0};
    glm::dvec3 C = {0, 0, 0};

    Triangle(glm::dvec3 a, glm::dvec3 b, glm::dvec3 c);
    [[nodiscard]] bool intersect(const Ray& ray, Hit& hit) const override;
    [[nodiscard]] BoundingBox boundingBox() const override;
    [[nodiscard]] glm::dvec3 normal() const;
    [[nodiscard]] glm::dvec2 texMapping(const glm::dvec3& intersect) const;
    void setTexCoords(glm::dvec2 ca, glm::dvec2 cb, glm::dvec2 cc);
    void invalidate();
};

class Sphere final : public Entity {
  public:
    glm::dvec3 center = {0, 0, 0};
    double radius = 0;

    Sphere();
    Sphere(glm::dvec3 center, double radius);
    [[nodiscard]] bool intersect(const Ray& ray, Hit& hit) const override;
    [[nodiscard]] BoundingBox boundingBox() const override;

  private:
    [[nodiscard]] glm::dvec2 texMapping(const glm::dvec3& intersect) const;
};
