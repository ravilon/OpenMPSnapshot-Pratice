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

#include "RandomUtils.h"
#include "Ray.h"
#include <glm/glm.hpp>

/// Represents the camera with information about the 'sensor' size.
class Camera {
    /// Location of the camera focus point.
    glm::dvec3 pos_;

    /// Normalized vector pointing from sensor to focus point.
    glm::dvec3 w_;
    /// Normalized vector pointing to the right.
    glm::dvec3 u_;
    /// Normalized vector pointing upwards.
    glm::dvec3 v_;

    /// Diagonal of the sensor
    const double sensor_diag_ = 0.035;

    /// Focal distance of the camera
    const double focal_dist_ = 0.04;

    /// Window width.
    double window_width_ = 0;
    /// Window height.
    double windows_height_ = 0;
    /// The factor by which the sensor is stretched to match the display size.
    double window_scale_ = 0.0;

  public:
    explicit Camera(glm::dvec3 pos);

    /**
     * Instantiates a new camera object with the given position and viewing direction
     * @param pos camera focal point position
     * @param look_at point of interest, center of the camera sensor
     * @param v_up z-axis of the camera
     */
    Camera(glm::dvec3 pos, glm::dvec3 look_at, glm::dvec3 v_up = {0, 0, 1});

    /**
     * Creates a ray that passes through the given pixel position.
     * @param x x-position
     * @param y y-position
     * @return randomized ray through this pixel
     */
    [[nodiscard]] Ray getRay(double x, double y) const;

    /**
     * Sets the cameras sensor resolution.
     * @param w resolution in x direction
     * @param h resolution in y direction
     */
    void setWindowSize(double w, double h);
};
