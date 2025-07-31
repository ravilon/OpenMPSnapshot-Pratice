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
#include <algorithm>
#include <vector>

class BVH : public Entity {
    class Node; // base class for the hierarchy tree
    class LeafNode;
    class InnerNode;

    /**
     * Nodes aren't split if they have less than this number of elements.
     */
    const size_t cutoff_size_;

    /**
     * Root of the volume hierarchy.
     */
    std::unique_ptr<Node> root_;

  public:
    /**
     * Creates a new bounding volume hierarchy from the given vector of triangles
     *
     * @param faces vector of faces for contained in the volume
     * @param cutoffSize maximum number of elements per node
     */
    explicit BVH(std::vector<Triangle> faces, size_t cutoffSize = 20);

    [[nodiscard]] BoundingBox boundingBox() const override;

    bool intersect(const Ray& ray, Hit& hit) const override;

    void setMaterial(std::shared_ptr<Material> material) override;

  private:
    /**
     * Creates a hierarchy from the given triangle list and returns the root node
     *
     * @param depth the entry depth of the subtree
     * @param faces triangles in the subtree
     * @return root node of the subtree
     */
    std::unique_ptr<Node> construct(size_t depth, std::vector<Triangle> faces);
};
