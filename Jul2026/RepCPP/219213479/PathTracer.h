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

#include <memory>
#include <random>

#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>

#include "Camera.h"
#include "Image.h"
#include "Octree.h"

class PathTracer {
    bool running_ = false;
    size_t samples_;
    Camera camera_;
    std::shared_ptr<const Octree> scene_;
    std::shared_ptr<Image> image_;

  public:
    PathTracer() = delete;
    explicit PathTracer(const Camera& camera, std::shared_ptr<const Octree> scene);

    void setScene(std::shared_ptr<const Octree> scene);
    void setSampleCount(size_t samples);
    void run(int w, int h);
    [[nodiscard]] bool running() const;
    void stop();
    void start();

    [[nodiscard]] std::shared_ptr<Image> getImage() const;

  private:
    /**
     * Iterative implementation of the path tracing.
     *
     * @param x x coordinate of the current pixel
     * @param y y coordinate of the current pixel
     * @return the light intensity transported on the traced path
     */
    [[nodiscard]] glm::dvec3 computePixel(int x, int y) const;

    /**
     * Recursive implementation of the path tracing.
     *
     * @param ray input ray
     * @return The light intensity transported on the traced path
     */
    [[nodiscard]] glm::dvec3 computePixel(const Ray& ray) const;
};
