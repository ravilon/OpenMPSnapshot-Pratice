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
#include "NDChecker.h"
#include <QtGui/QImage>
#include <array>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <utility>

/**
 * Base class for all textures. Provides a function to obtain the texture color for a given set of
 * texture coordinates.
 */
class Texture {
  public:
    virtual ~Texture() = default;

    /**
     * Returns the value of the texture at the given position.
     * @param uv position in the texture
     * @return color at the given position.
     */
    [[nodiscard]] virtual glm::dvec3 value(glm::dvec2 uv) const = 0;
};

/**
 * This class represents a texture which is uniform in color at every position.
 */
class ConstantTexture final : public Texture {
    glm::dvec3 color_;

  public:
    constexpr explicit ConstantTexture(const glm::dvec3 color) : color_(color) {}

    [[nodiscard]] glm::dvec3 value(glm::dvec2 uv) const override { return color_; }
};

/**
 * This class is a container for two other textures. The other textures are arranged, such that
 * they form a checkerboard. The number of squares can be set freely.
 */
class CheckerboardMaterial final : public Texture {
    const NdChecker<2> checker_;
    const std::shared_ptr<ConstantTexture> color1_;
    const std::shared_ptr<ConstantTexture> color2_;

  public:
    explicit CheckerboardMaterial(const size_t squares = 10,
                                  std::shared_ptr<ConstantTexture> color1 =
                                      std::make_shared<ConstantTexture>(glm::dvec3(0, 0, 0)),
                                  std::shared_ptr<ConstantTexture> color2 =
                                      std::make_shared<ConstantTexture>(glm::dvec3(1, 1, 1)))
        : checker_(squares), color1_(std::move(color1)), color2_(std::move(color2))
    {
    }

    [[nodiscard]] glm::dvec3 value(const glm::dvec2 uv) const override
    {
        return checker_.at({uv.x, uv.y}) ? color2_->value(uv) : color1_->value(uv);
    }
};

/**
 * This texture sources its colors from an image file. The returned color is that of the nearest
 * pixel.
 */
class ImageBackedTexture final : public Texture {
    size_t width;
    size_t height;
    std::vector<glm::dvec3> image;

  public:
    explicit ImageBackedTexture(const std::string& name) : width(0), height(0)
    {
        QImage img;
        if (!img.load(QString::fromStdString(name))) {
            std::cerr << "Could not load texture at location " << name << "." << std::endl;
            throw std::runtime_error("Could not load texture at location " + name + ".");
        }
        width = img.size().width();
        height = img.size().height();
        image.reserve(width * height);

        for (int y = static_cast<int>(height - 1); y >= 0; --y) {
            for (int x = 0; x < width; ++x) {
                const auto p = img.pixel(x, y);
                image.emplace_back(qRed(p) / 255., qGreen(p) / 255., qBlue(p) / 255.);
            }
        }
    }
    [[nodiscard]] glm::dvec3 value(glm::dvec2 uv) const override
    {
        // FIXME: map texture coordinates to [0,1] otherwise -> out of bounds
        const auto x = lround(uv.x * (static_cast<double>(width) - 1.0));
        const auto y = lround(uv.y * (static_cast<double>(height) - 1.0));
        return image[y * width + x];
    }
};
