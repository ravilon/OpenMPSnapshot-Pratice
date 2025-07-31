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
#include "Ray.h"
#include "Texture.h"
#include <glm/glm.hpp>

/**
 * \brief Abstract base class for all materials.
 * The class describes the light emitted as well as the scattering behavior of the material.
 */
class Material {
  public:
    virtual ~Material() = default;

    /**
     * \brief Computes a new scattered ray given an input and hit.
     * \param in The incoming ray that hit the material
     * \param ir intersection record describing the hit properties
     * \param attenuation outputs the damping of the light transport for the scattered ray
     * \param scatter_ray the new scattered ray
     * \return true if there is a scattered ray
     */
    virtual bool scatter(const Ray& in,
                         const Hit& ir,
                         glm::dvec3& attenuation,
                         Ray& scatter_ray) const = 0;

    /**
     * \brief Returns the emission of the given material
     * \param uv uv coordinates of of the emission position
     * \return a color vector describing the emission per channel
     */
    [[nodiscard]] virtual glm::dvec3 emission(const glm::dvec2& uv) const
    {
        return glm::dvec3(0, 0, 0);
    }
};

/**
 * \brief Material that has Lambertian scattering properties.
 *
 * The material scatters the rays into the hemisphere uniformly.
 */
class LambertianMaterial : public Material {
  protected:
    std::shared_ptr<Texture> tex_;

  public:
    explicit LambertianMaterial(const glm::dvec3& color);
    explicit LambertianMaterial(std::shared_ptr<Texture> tex);
    bool scatter(const Ray& in,
                 const Hit& ir,
                 glm::dvec3& attenuation,
                 Ray& scatter_ray) const override;
};

/**
 * \brief This material acts as a diffuse light source with the given color or texture as light
 * intensity.
 */
class DiffuseLight final : public LambertianMaterial {
  public:
    explicit DiffuseLight(const glm::dvec3& color);
    explicit DiffuseLight(std::shared_ptr<Texture> tex);
    [[nodiscard]] glm::dvec3 emission(const glm::dvec2& uv) const override;
};

/**
 * \brief This material has properties similar to a metal.
 *
 * The material acts as a reflector for low values of spec_size_ and more like a diffuse object for
 * high values.
 */
class MetalLikeMaterial final : public Material {
    glm::dvec3 attenuation_;
    double spec_size_;

  public:
    MetalLikeMaterial(const glm::dvec3& attenuation, double spec_size);
    bool scatter(const Ray& in,
                 const Hit& ir,
                 glm::dvec3& attenuation,
                 Ray& scatter_ray) const override;
};

/**
 * \brief Material with dielectric properties.
 *
 * The material behaves similar to glass, i.e. it reflects and refracts light. The ratio of
 * reflected to refracted light is governed by the fresnel equation.
 */
class Dielectric final : public Material {
    double refractive_index_;

  public:
    explicit Dielectric(double refractive_index);
    bool scatter(const Ray& in,
                 const Hit& ir,
                 glm::dvec3& attenuation,
                 Ray& scatter_ray) const override;

  private:
    /**
     * Computes the probability of a ray reflecting instead of refracting. The computation is done
     * with Schlick's approximation.
     * @return probability of a ray reflecting
     */
    static double reflectance_schlick(double n1, double n2, double cosI);

    /**
     * Computes the probability of a ray reflecting instead of refracting. The computation is done
     * with Fresnel's equation. The light is seen a unpolarized and hence the two horizontal and
     * vertical components are averaged.
     * @return probability of a ray reflecting
     */
    static double reflectance_fresnel(double n1, double n2, double cosI, double cosT);
};
