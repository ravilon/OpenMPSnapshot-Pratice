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

#include "Material.h"
#include "ObjReader.h"
#include "Octree.h"
#include "entities.h"
#include <filesystem>
#include <memory>
#include <utility>

enum class SceneSetting { Empty, Cornell, Exam, Pig, Cow, Dragon };

class Scene {
    constexpr static glm::dvec3 black = glm::dvec3(0, 0, 0);
    constexpr static glm::dvec3 white = glm::dvec3(1, 1, 1);
    constexpr static glm::dvec3 red = glm::dvec3(1, 0, 0);
    constexpr static glm::dvec3 green = glm::dvec3(0, 1, 0);
    constexpr static glm::dvec3 blue = glm::dvec3(0, 0, 1);
    constexpr static glm::dvec3 yellow = glm::dvec3(1, 1, 0);
    constexpr static glm::dvec3 cyan = glm::dvec3(0, 1, 1);
    constexpr static glm::dvec3 magenta = glm::dvec3(1, 0, 1);

    constexpr static const char* pig_body_obj_ = "pig_body.obj";
    constexpr static const char* pig_eyes_obj_ = "pig_eyes.obj";
    constexpr static const char* pig_pupils_obj_ = "pig_pupils.obj";
    constexpr static const char* pig_tongue_obj_ = "pig_tongue.obj";
    constexpr static const char* cow_obj_ = "spot_triangulated.obj";
    constexpr static const char* cow_tex_ = "spot_texture.png";
    constexpr static const char* dragon_obj_ = "dragon-3.obj";

    std::filesystem::path share_dir_;
    std::vector<std::unique_ptr<Entity>> entities_;
    std::shared_ptr<Octree> tree_;

  public:
    /**
     * Constructs an empty scene with the given dimensions and resource directory.
     * @param shareDir the directory where textures and model files are stored
     * @param min the minimal coordinates of the scene
     * @param max the maximal coordinates of the scene
     */
    Scene(std::filesystem::path shareDir, glm::dvec3 min, glm::dvec3 max);

    /**
     * Sets the scene to a predefined setting.
     * @param setting setting specification
     */
    void useSceneSetting(SceneSetting setting);

    /**
     * Adds a Cornell box to the scene.
     * @return this scene
     */
    Scene& addCornellBox(double side_len = 6.0,
                         double light_size = 5.0,
                         double light_intensity = 2.5);

    /**
     * Adds a diffuse cube and a glass and metal sphere.
     * @return this scene
     */
    Scene& addCornellContent();

    /**
     * Adds the scene used for the final examination.
     * @return this scene
     */
    Scene& addExamRender();

    /**
     * Adds colored indicators for the three axis and their directions.
     * @return this scene
     */
    Scene& addAxisIndicator();

    /**
     * Adds a pig to the scene.
     * @return this scene
     */
    Scene& addPig(glm::dvec3 rotate = {-glm::pi<double>() / 2, 0.0, -glm::pi<double>() / 3},
                  double scale = 3,
                  glm::dvec3 translation = {0, 0, -1},
                  bool add_box = true);

    /**
     * Adds the Stanford dragon to the scene.
     * @return this scene
     */
    Scene& addDragon();

    /**
     * Adds Spot the cow to the scene.
     * @return this scene
     */
    Scene& addCow();

    /**
     * Removes all entities from the scene.
     */
    void clear();

    /**
     * Returns the scene contents.
     * @return Octree with all scene entities
     */
    std::shared_ptr<Octree> getTree();

  private:
    /**
     * Searches for the specified relative file name in the resource directory. If the function
     * returns the file exists.
     * @param relative_name desired file name
     * @return the found file path
     * @throws std::runtime_error if the file is not found
     */
    [[nodiscard]] std::filesystem::path resolveFile(const std::string& relative_name) const;

    /**
     * Adds an entity to the scene.
     * @param entity the entity to add
     */
    void insert(std::unique_ptr<Entity> entity);
};
