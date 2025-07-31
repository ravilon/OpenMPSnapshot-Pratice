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

#include "PathTracer.h"
#include "Material.h"
#include "entities.h"
#include <chrono>
#include <iostream>

PathTracer::PathTracer(const Camera& camera, std::shared_ptr<const Octree> scene)
    : samples_(2048), camera_(camera), scene_(std::move(scene)),
      image_(std::make_shared<Image>(0, 0))
{
}

void PathTracer::setScene(std::shared_ptr<const Octree> scene) { scene_ = std::move(scene); }

void PathTracer::setSampleCount(const size_t samples) { samples_ = samples; }

void PathTracer::run(const int w, const int h)
{
    const auto samples = samples_;

    std::vector<glm::dvec3> buffer;
    buffer.reserve(w * h);
    for (auto i = 0; i < w * h; i++) {
        buffer.emplace_back(0, 0, 0);
    }

    image_ = std::make_shared<Image>(w, h);
    camera_.setWindowSize(w, h);
    // The structure of the for loop should remain for incremental rendering.
    for (auto s = 1; s <= samples; ++s) {
        if (!running_) {
            return;
        }
        std::cout << "Sample " << s << std::endl;
#pragma omp parallel for schedule(dynamic, 1)
        for (auto y = 0; y < h; ++y) {
            for (auto x = 0; x < w; ++x) {
                if (running_) {
                    const auto color = computePixel(x, y);
#pragma omp critical
                    {
                        const auto pos = x * h + y;
                        buffer[pos] += color;
                        const auto pix = buffer[pos] / static_cast<double>(s);
                        image_->setPixel(x, y, glm::clamp(pix, 0.0, 1.0));
                    }
                }
            }
        }
    }
}

glm::dvec3 PathTracer::computePixel(const int x, const int y) const
{
    constexpr auto max_bounces = 5;

    // the currently active ray
    auto ray = camera_.getRay(x, y);
    // the total amount of light carried over this path
    auto light = glm::dvec3(0, 0, 0);
    // value gives the amount of light that is carried per color channel over the path
    auto throughput = glm::dvec3(1, 1, 1);

    for (auto i = 0; i < max_bounces; i++) {
        Hit hit;
        if (!scene_->intersect(ray, hit)) {
            break; // the ray didn't hit anything -> no contribution.
        }

        // add light reduced by combined attenuation
        light += throughput * hit.mat->emission(hit.uv);

        glm::dvec3 bounce_attenuation;
        auto scatter_ray(ray);
        if (!hit.mat->scatter(ray, hit, bounce_attenuation, scatter_ray)) {
            break; // the ray did not scatter -> no further contribution
        }
        ray = scatter_ray;
        throughput *= bounce_attenuation;
    }

    return light;
}

glm::dvec3 PathTracer::computePixel(const Ray& ray) const
{
    Hit hit;
    if (!scene_->intersect(ray, hit)) {
        return {0, 0, 0};
    }

    const auto light = hit.mat->emission(hit.uv);

    if (ray.child_level <= 5) {
        glm::dvec3 attenuation;
        auto scatter_ray(ray);
        if (hit.mat->scatter(ray, hit, attenuation, scatter_ray)) {
            return light + attenuation * computePixel(scatter_ray);
        }
    }

    return light;
}

bool PathTracer::running() const { return running_; }

void PathTracer::stop() { running_ = false; }

void PathTracer::start() { running_ = true; }

std::shared_ptr<Image> PathTracer::getImage() const { return image_; }
