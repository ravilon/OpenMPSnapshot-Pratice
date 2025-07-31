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
#include "Entity.h"
#include <array>
#include <glm/glm.hpp>
#include <memory>
#include <utility>
#include <vector>

class Octree : public Hittable {
    class Node;
    std::unique_ptr<Node> root_;

  public:
    /**
     * Constructs a new octree with the given min and max for the bbox of the tree.
     *
     * @param min minimum position of entities in the tree
     * @param max maximum position of elements in the tree
     */
    Octree(glm::dvec3 min, glm::dvec3 max);

    /**
     * Store an entity in the correct position of the octree.
     *
     * @param object
     */
    void insert(Hittable* object) const;

    /**
     * Finds the closes intersecting entity in the tree. It is ensured that the hit occurred in
     * front of the ray, i.e. not behind the origin.
     *
     * @param ray intersecting ray
     * @param hit this record is filled with hit info, otherwise the data is invalid
     * @return true if a hit occurred
     */
    [[nodiscard]] bool intersect(const Ray& ray, Hit& hit) const override;

    /**
     * Returns the bounding box spanning the entire tree.
     *
     * @return bbox of the tree
     */
    [[nodiscard]] BoundingBox boundingBox() const override;

    /**
     * Deletes the content of the tree.
     */
    void clear();

    /**
     * Prints the tree in a lisp-like parenthesis-grouped style.
     *
     * @param o output stream
     * @param t tree object
     * @return output stream
     */
    friend std::ostream& operator<<(std::ostream& o, const Octree& t);
};
