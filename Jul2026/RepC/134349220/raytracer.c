/*
Copyright (c) 2012, Shiben Bhattacharjee, Naveen Kumar

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom
the Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
*/

#include "types.h"
#include "scene.h"
#include   "ray.h"
#include "shade.h"
#include "image.h"
#include "omp.h"
#include "time.h"

int tid;
int nth;



void createScene(Scene *scene)
{
    Object cone, cylinder, cube, sphere, plane_base, plane_right, plane_left, plane_back;
    Material mtl_matt1, mtl_matt2, mtl_matt3, mtl_matt4, mtl_glass, mtl_shiny1, mtl_shiny2;
    Vector red, blue, green, purple, white;

    /* lets collect a few colors */
    setVector(&red,    1.0f, 0.0f, 0.0f);
    setVector(&green,  0.0f, 1.0f, 0.0f);
    setVector(&blue,   0.0f, 0.0f, 1.0f);
    setVector(&purple, 1.0f, 0.0f, 1.0f);
    setVector(&white,  1.0f, 1.0f, 1.0f);

    /* Create some materials */
    /* setMaterial(&mtl, color, kambient, kdiffuse, kspecular, shininess, refl, refr, ir) */
    setMaterial(&mtl_shiny1, white,  0.1f, 0.5f, 0.4f, 2.0f,  0.4f, 0.2f, 1.4f);
    setMaterial(&mtl_shiny2, blue,   0.1f, 0.5f, 0.4f, 3.0f,  0.2f, 0.0f, 1.4f);
    setMaterial(&mtl_matt1,  red,    0.1f, 0.5f, 0.4f, 32.0f, 0.0f, 0.0f, 1.4f);
    setMaterial(&mtl_matt2,  blue,   0.1f, 0.5f, 0.4f, 2.0f,  0.0f, 0.0f, 1.4f);
    setMaterial(&mtl_matt3,  green,  0.1f, 0.5f, 0.4f, 2.0f,  0.0f, 0.0f, 1.4f);
    setMaterial(&mtl_matt4,  purple, 0.1f, 0.5f, 0.4f, 2.0f,  0.0f, 0.0f, 1.4f);
    setMaterial(&mtl_glass,  white,  0.0f, 0.5f, 0.0f, 0.0f,  0.0f, 0.9f, 1.4f);

    /* create some objects, attach created materials to them */
    createCone(     &cone,          mtl_matt4,  1.0f, 1.5f, 32);
    createSphere(   &sphere,        mtl_matt1,  0.9f, 3);
    createCylinder( &cylinder,      mtl_shiny2, 0.75f, 1.0f, 32);
    createCube(     &cube,          mtl_glass,  1.0f);
    createPlaneXZ(  &plane_base,    mtl_shiny1, 10.0f);
    createPlaneXZ(  &plane_left,    mtl_matt1,  10.0f);
    createPlaneXZ(  &plane_right,   mtl_matt2,  10.0f);
    createPlaneXZ(  &plane_back,    mtl_matt3,  10.0f);

    /* arrange them in the scene */
    transformObject(&cone,     translate(2.0f, 0.0f, -2.8f));
    //transformObject(&cone,     matMatMult(translate(2.0f, 0.0f, -2.8f), rotateY(45.0f)));
    //transformObject(&cone,     scale(0.2f,0.2f,0.2f));
    transformObject(&cube,     translate(2.0f, 0.5f, -1.0f));
    //transformObject(&cube,     matMatMult(translate( 2.0f, 0.5f, -1.0f), rotateY(45.0f)));
    //transformObject(&cube,     scale(0.5f,0.5f,0.5f));
    //transformObject(&cylinder, translate(-0.0f, 0.0f, -1.0f));
    //transformObject(&sphere,   translate(-0.0f, 0.9f, -2.7f));
    //transformObject(&plane_base,  translate( 1.0f, 0.0f, -4.0f));
    transformObject(&plane_base,  translate( 1.0f, 0.0f, -4.0f));
    //transformObject(&plane_left,  matMatMult(translate( -2.0f, 0.0f, -4.0f), rotateZ(-90.0)));
    //transformObject(&plane_right, matMatMult(translate(  4.0f, 0.0f, -4.0f), rotateZ(90.0)));
    //transformObject(&plane_back,  translate(  1.0f, 0.0f, -6.0f));
    transformObject(&plane_back,  matMatMult(translate(  1.0f, 0.0f, -6.0f), rotateX(90.0)));

    /* add them to the scene */
    initScene(scene, 4);
    addObjectToScene(scene, cone);
    addObjectToScene(scene, cube);
    //addObjectToScene(scene, cylinder);
    //addObjectToScene(scene, sphere);
    addObjectToScene(scene, plane_base);
    //addObjectToScene(scene, plane_right);
    //addObjectToScene(scene, plane_left);
    addObjectToScene(scene, plane_back);
}

Vector trace(Ray ray, Scene scene, Light light, int recur)
{
    Hit hit;
    Vector output, reflColor, refrColor;
    float krefl, krefr;

    /* default return color */
    setVector(&output, 0.0f, 0.0f, 0.0f);

    hit = intersectScene(ray, scene);

    if(hit.objid >= 0)
    {
        /* current object color */
        output = add(ambient(hit, scene, light), add(diffuse(hit, scene, light), specular(hit, scene, light)));

        if(recur > 0)
        {
            /* collect color captured by reflected ray */
            reflColor = trace(reflectRay(hit), scene, light, recur - 1);
            krefl = scene.obj[hit.objid].material.reflectivity;
            output = add(output, floatVecMult(krefl, reflColor));

            /* collect color captured by refracted ray */
            refrColor = trace(refractRay(hit, scene.obj[hit.objid].material.ir), scene, light, recur - 1);
            krefr = scene.obj[hit.objid].material.translucency;
            output = add(output, floatVecMult(krefr, refrColor));
        }

        /* reduce color by the shadow factor of the light */
        return floatVecMult(1.0f - traceShadow(hit, scene, light), output);
    }

    return output;
}

int main(int argc, char **argv)
{
    int width = 600, height =400, recur = 2, i, j;
    char *filename = "output.ppm";
    //double exetime = 0;

  	//srand(time(NULL));

    Vector lcolor, lpos, camPos, lookat;
    Light light;
    Camera cam;
    Hit hit;
    Ray ray;
    Color outcolor;
    Image image;
    Scene scene;

    /* setup scene */
    createScene(&scene);

    //exetime = omp_get_wtime();

    /* setup light */
    setVector(&lcolor, 1.0f, 1.0f, 1.0f);
    setVector(&lpos,  -1.0f, 4.0f, 4.0f);
    setLight(&light, lpos, lcolor, 0.3f);

    /* setup camera */
    setVector(&camPos, 1.0f, 2.0f,  4.0f);
    setVector(&lookat, 1.0f, 0.0f, -6.0f);
    setCamera(&cam, camPos, lookat, 45.0f, width, height);

    /* setup image */
    initImage(&image, width, height);

    printf("Image: %s, Size: %dx%d\n", filename, width, height);
    printf("Initial Ray Count: %d, Triangle Count: %d, Recursion Depth: %d\n", width * height, getTriangleCount(scene), recur);
    printf("Raytracing...\n");
    int index=0;
    /* tracing */
    #pragma omp parallel for collapse(2) private(outcolor, ray, i, j, index) firstprivate(width, height, image) shared(scene, light, recur)
      for(j = 0; j < height; j++){
          for(i = 0; i < width; i++)
          {
              //printf(" ray id: %d , %d\n",i,j);
              ray = generateRay(i, j, cam);
              outcolor = vector2color(trace(ray, scene, light, recur));
              #pragma omp critical
                setPixel(&image, i, j, outcolor, index, width, height);
                //printf("finish  critical section: %d \n", omp_get_thread_num());
          }
      }





    printf("Done, writing image...\n");
    writeImage(image, filename);
    cleanScene(&scene);
    //printf("Elapse time: %lf\n", exetime);
    printf("Done\n");
}
