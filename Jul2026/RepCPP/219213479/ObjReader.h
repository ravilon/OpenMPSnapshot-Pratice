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

#include "BVH.h"
#include "Entity.h"
#include <vector>

namespace obj {

using ObjContent = std::vector<Triangle>;

/**
 * Creates a triangle list of an explicit sphere. The algorithm is quite simple. First a starting
 * shape is produced. This can either be a Tetrahedron (four equilateral triangles) or an
 * icosahedron (20 equilateral triangles). Until the maximum number of subdivision is reached each
 * triangle is subdivided. The subdivision computes the center of each triangle side. These centers
 * are projected to the implicit sphere with the given radius. Then the triangle is replaced with
 * four new triangles formed from the three center points and three corner points.
 * @param center sphere center
 * @param radius sphere radius
 * @param sub_divisions number of subdivisions of each triangle
 * @param use_tetrahedron start with a tetrahedron or icosahedron
 * @return list of triangles of the sphere
 */
[[nodiscard]] ObjContent makeSphere(glm::dvec3 center = {0, 0, 0},
                                    double radius = 1,
                                    int sub_divisions = 2,
                                    bool use_tetrahedron = false);

/**
 * This function build a quad with the given corner points. The corner points must be in ccw order,
 * such that the implicit normals are correct.
 * @return quad build out of two triangles
 */
[[nodiscard]] ObjContent makeQuad(glm::dvec3 a, glm::dvec3 b, glm::dvec3 c, glm::dvec3 d);

/**
 * The function creates an axis-aligned cube with the given side length and center.
 * @param center cube center
 * @param side_length cube side length
 * @return list of triangles of the cube
 */
[[nodiscard]] ObjContent makeCube(glm::dvec3 center, double side_length);

/**
 * This function is similar to makeCube but the side lengths might differe per dimension.
 *
 * @param center object center point
 * @param size side length per dimension
 * @return list of triangles of the cube
 */
[[nodiscard]] ObjContent makeCuboid(glm::dvec3 center, glm::dvec3 size);

/**
 * This function creates a fully unrestricted object with four corner points.
 * @param corners array of corner points
 * @return the triangles of the created object
 */
[[nodiscard]] ObjContent makeOctet(std::array<glm::dvec3, 8> corners);

/**
 * Builds a cone from the given parameters. The constuction starts at the center and builds a circle
 * perpendicular to the height axis. Then a number of points is selected uniformly distributed on
 * the circles border. These points are used to form triangles between circle center and border and
 * tip and border.
 * @param center center point of the cone base
 * @param tip cone tip
 * @param radius cone radius
 * @param slices number of points on the border of the cone base
 * @return list of cone triangles
 */
[[nodiscard]] ObjContent makeCone(glm::dvec3 center, glm::dvec3 tip, double radius, size_t slices);

/**
 * Reads a wavefront obj file and creates a triangle list from it. The same restrictions as in
 * readObjStream(stream,list) apply.
 * @param file file name
 * @return list of triangles in the file
 */
[[nodiscard]] ObjContent readObjFile(const std::string& file);

/**
 * Reads a wavefront obj file from the provided stream. The same restrictions as in
 * readObjStream(stream,list) apply.
 * @param is input stream
 * @return list of triangles in the stream
 */
[[nodiscard]] ObjContent readObjStream(std::istream& is);

/**
 * Reads a wavefront obj file from the provided stream. The reader is quite simple and
 * does not respect material, grouping or smoothing parameters.
 * @param is input stream
 * @param content list of triangles in the stream
 */
void readObjStream(std::istream& is, ObjContent& content);

std::ostream& operator<<(std::ostream& os, const ObjContent& content);

std::istream& operator>>(std::istream& is, ObjContent& content);

/**
 * Computes the AABB with tightest fit enclosing all triangles in the list
 * @param content list of triangles
 * @return bbox containing all triangles
 */
BoundingBox computeBBox(const ObjContent& content);

class Transform {
    class Step;
    class Translate;
    class Scale;
    class Center;
    class Rotate;
    std::vector<std::unique_ptr<Step>> transforms_;

  public:
    Transform& rotate_x(double angle);
    Transform& rotate_y(double angle);
    Transform& rotate_z(double angle);
    Transform& center();
    Transform& translate(glm::dvec3 delta);
    Transform& scale(double scale);
    Transform& scale(glm::dvec3 scale);

    [[nodiscard]] ObjContent apply(ObjContent content) const;
    [[nodiscard]] std::unique_ptr<BVH> to_bvh(ObjContent content) const;
    [[nodiscard]] std::unique_ptr<BVH> to_bvh(std::string file) const;

  private:
    class Step {
        virtual void pre(const ObjContent& content) {}
        virtual void process(Triangle& t) = 0;

        friend ObjContent Transform::apply(ObjContent content) const;
    };
};

} // namespace obj
