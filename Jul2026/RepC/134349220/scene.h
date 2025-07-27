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

#ifndef _SCENE_H_
#define _SCENE_H_

#include "types.h"
#include "omp.h"

#define THREADS 8;

typedef struct Material
{
    Vector  color;          /* Color of the object */
    float   kambient;       /* Ambient constant */
    float   kdiffuse;       /* Diffuse constant */
    float   kspecular;      /* Specular constant */
    float   shininess;      /* Exponent of specular component */
    float   reflectivity;   /* Amount of reflectivity of the surface, 0.0f means a matt object, 1.0f means a mirror */
    float   translucency;   /* Amount of translucency of the surface, 0.0f means opaque object, 1.0f means a Saint Gobain's glass */
    float   ir;             /* Index of refraction of a translucent object */
}
Material;

typedef struct Object
{
    Material  material;     /* Material properties of this object */
    int       ntris;        /* Number of Triangles */
    Triangle  *tri;         /* Triangle list of the object */
}
Object;

typedef struct Scene
{
    int     nObjects;       /* Number of objects in the scene */
    int     nMax;           /* Max number of objects scene can hold */
    Object  *obj;           /* Object list of the Scene */
}
Scene;

typedef struct Light
{
    Vector  position;
    Vector  color;          /* Light color, (1.0f, 1.0f, 1.0f) in most cases */
    float   shadow;         /* Shadow factor, 0.0f means no shadow, 1.0f means solid black shadow */
}
Light;

typedef struct Camera
{
    Vector position;
    Vector look;            /* Look unit vector */
    Vector hori;            /* Horizontal unit vector */
    Vector vert;            /* Vertical unit vector */
    Vector up;              /* General Up unit vector (0,1,0) */
    float  fov2;            /* Half of Vertical fov in radians */
    float  ar;              /* Aspect ratio */
    int    width;
    int    height;
}
Camera;


/* Data type setters */
void setMaterial(Material *material, Vector color, float kambient, float kdiffuse, float kspecular, float shininess, float reflectivity, float translucency, float ir)
{
    (*material).color        = color;
    (*material).kambient     = kambient;
    (*material).kdiffuse     = kdiffuse;
    (*material).kspecular    = kspecular;
    (*material).shininess    = shininess;
    (*material).reflectivity = reflectivity;
    (*material).translucency = translucency;
    (*material).ir           = ir;
}

void setObject(Object *object, Material material, int ntris, Triangle *tri)
{
    (*object).material = material;
    (*object).ntris    = ntris;
    (*object).tri      = tri;
}

void initScene(Scene *scene, int nMax)
{
    /* Only allocates memory, use addObjectToScene() to populate the scene */
    (*scene).nObjects = 0;
    (*scene).nMax     = nMax;
    (*scene).obj      = (Object *)malloc(sizeof(Object) * nMax);
}

void addObjectToScene(Scene *scene, Object object)
{
    if((*scene).nObjects < (*scene).nMax)
    {
        (*scene).obj[ (*scene).nObjects ] = object;
        (*scene).nObjects++;
    }
    else
        fprintf(stderr, "Error: Maximum number of objects reached\n");
}

int getTriangleCount(Scene scene)
{
    int i, count = 0;
    for(i = 0; i < scene.nObjects; i++)
        count += scene.obj[i].ntris;
    return count;
}

void setLight(Light *light, Vector position, Vector color, float shadow)
{
    (*light).position = position;
    (*light).color    = color;
    (*light).shadow   = shadow;
}

void setCamera(Camera *cam, Vector position, Vector lookat, float fov, int width, int height)
{
    Vector look, hori, vert, up;

    /* Camera assumes up Vector is (0,1,0) */
    setVector(&up, 0.0f, 1.0f, 0.0f);
    look = normalize(subtract(lookat, position));
    hori = cross(look, up);
    vert = cross(hori, look);

    (*cam).position = position;
    (*cam).look     = look;
    (*cam).hori     = hori;
    (*cam).vert     = vert;
    (*cam).up       = up;

    (*cam).fov2     = deg2rad(fov * 0.5f);
    (*cam).ar       = (float)width / (float)height;
    (*cam).width    = width;
    (*cam).height   = height;
}


/* Deleting objects */
void cleanObject(Object *object)
{
    free((*object).tri);

    (*object).tri   = NULL;
    (*object).ntris = 0;
}

void cleanScene(Scene *scene)
{
    int i;
    for(i = 0; i < (*scene).nMax; i++)
        cleanObject( &(*scene).obj[i] );

    free((*scene).obj);

    (*scene).obj      = NULL;
    (*scene).nObjects = 0;
    (*scene).nMax     = 0;
}


/* Object creators */
void subdivide(Triangle src, Triangle *dest, float radius)
{
    Triangle mid;
    Vector mid0, mid1, mid2;

    /* mid points of the edges of the triangle */
    mid0 = floatVecMult(radius, normalize(add(src.v0, src.v1)));
    mid1 = floatVecMult(radius, normalize(add(src.v1, src.v2)));
    mid2 = floatVecMult(radius, normalize(add(src.v2, src.v0)));

    /* input triangle subdivided into 4 based on the edges mid point */
    setTriangle(&dest[0], mid0,   mid1,   mid2);
    setTriangle(&dest[1], src.v0, mid0,   mid2);
    setTriangle(&dest[2], mid0,   src.v1, mid1);
    setTriangle(&dest[3], mid2,   mid1,   src.v2);
}
void subdivideRecursively(int recur, Triangle in, Triangle *out, int outi, float radius)
{
    Triangle t[4];
    /* span is the amount subdivide should jump so that final triangles are placed fine */
    int i, span = (int)pow(4.0f, (float)(recur - 1));

    /* if we are still recursing, subdivide further */
    if(recur)
    {
        /* in triangle subdivided, and those are further sent for subdivision */
        subdivide(in, t, radius);
        for(i = 0; i < 4; i++)
            subdivideRecursively(recur - 1, t[i], out, outi + i * span, radius);
    }
    else
        /* recursion has ended, lets populate the incoming triangle to *out */
        out[outi] = in;
}


void createSphere(Object *object, Material material, float radius, int resolution)
{
    /* resolution is power of 4, can be zero and more */
    Vector v[6];
    Triangle tri[8];
    Triangle *out;

    /* Sphere starts as an ochedron, minimum triangles is 8 and subdivides to 4 each time */
    int i, fres = 8 * (int)pow(4.0f, (float)resolution);
    out = (Triangle *)malloc(sizeof(Triangle) * fres);

    setVector(&v[0], 0.0f, 0.0f, -radius);
    setVector(&v[1], 0.0f, -radius, 0.0f);
    setVector(&v[2], -radius, 0.0f, 0.0f);
    setVector(&v[3], 0.0f, 0.0f,  radius);
    setVector(&v[4], 0.0f,  radius, 0.0f);
    setVector(&v[5],  radius, 0.0f, 0.0f);

    /* Octahedron's top triangles */
    setTriangle(&tri[0], v[5], v[4], v[3]);
    setTriangle(&tri[1], v[3], v[4], v[2]);
    setTriangle(&tri[2], v[2], v[4], v[0]);
    setTriangle(&tri[3], v[0], v[4], v[5]);

    /* Octahedron's bottom triangles */
    setTriangle(&tri[4], v[3], v[1], v[5]);
    setTriangle(&tri[5], v[5], v[1], v[0]);
    setTriangle(&tri[6], v[0], v[1], v[2]);
    setTriangle(&tri[7], v[2], v[1], v[3]);

    /* Lets start subdividing these 8 triangles recursively */
    for(i = 0; i < 8; i++)
        subdivideRecursively(resolution, tri[i], out, i * fres / 8, radius);

    /* finally set the object */
    setObject(object, material, fres, out);
}

void createCone(Object *object, Material material, float radius, float height, int resolution)
{
    /* resolution is linear sampling here, min can be 4 */
    int i;
    float t1, t2;
    Vector v0, v1, top, origin;
    Triangle *tris;

    /* 2 times the resolution because we wan't to cover the base as well */
    tris = (Triangle *)malloc(sizeof(Triangle) * 2 * resolution);

    setVector(&top,    0.0f, height, 0.0f);
    setVector(&origin, 0.0f, 0.0f,   0.0f);

    for(i = 0; i < resolution; i++)
    {
        t1 = (float)i / (float)resolution;
        t1 = deg2rad(t1 * 360.0f);
        t2 = (float)(i + 1) / (float)resolution;
        t2 = deg2rad(t2 * 360.0f);

        setVector(&v0, radius * cos(t1), 0.0f, radius * sin(t1));
        setVector(&v1, radius * cos(t2), 0.0f, radius * sin(t2));

        /* Cone structure */
        setTriangle(&tris[2 * i + 0], top, v1, v0);

        /* Cone base */
        setTriangle(&tris[2 * i + 1], v1, origin, v0);
    }

    /* finally set the object */
    setObject(object, material, 2 * resolution, tris);
}

void createPlaneXZ(Object *object, Material material, float size)
{
    Triangle *tris;
    Vector v0, v1, v2;
    tris = (Triangle *)malloc(sizeof(Triangle)*2);

    int side;
    side = size * 0.5f;
    setVector(&v0, side, 0.0, side);
    setVector(&v1, side, 0.0, -side);
    setVector(&v2, -side, 0.0, -side);
    setTriangle(&tris[0], v0, v1, v2);

    setVector(&v0, side, 0.0, side);
    setVector(&v1, -side, 0.0, side);
    setVector(&v2, -side, 0.0, -side);
    setTriangle(&tris[1], v2, v1, v0);

    setObject(object, material, 2, tris);
}

void createCube(Object *object, Material material, float side)
{
    Vector v0, v1, v2, min, max;
    Triangle *tris;

    side = side * 0.5f;
    setVector(&min, -side, -side, -side);
    setVector(&max,  side,  side,  side);

    /* 6 sides, each side with 2 triangles */
    tris = (Triangle *)malloc(sizeof(Triangle) * 12);

    /* front */
    setVector(&v0, min.x, min.y, min.z);
    setVector(&v1, max.x, min.y, min.z);
    setVector(&v2, max.x, max.y, min.z);
    setTriangle(&tris[0], v2, v1, v0);

    setVector(&v0, max.x, max.y, min.z);
    setVector(&v1, min.x, max.y, min.z);
    setVector(&v2, min.x, min.y, min.z);
    setTriangle(&tris[1], v2, v1, v0);

    /* right */
    setVector(&v0, max.x, min.y, min.z);
    setVector(&v1, max.x, min.y, max.z);
    setVector(&v2, max.x, max.y, max.z);
    setTriangle(&tris[2], v2, v1, v0);

    setVector(&v0, max.x, max.y, max.z);
    setVector(&v1, max.x, max.y, min.z);
    setVector(&v2, max.x, min.y, min.z);
    setTriangle(&tris[3], v2, v1, v0);

    /* back */
    setVector(&v0, max.x, min.y, max.z);
    setVector(&v1, min.x, min.y, max.z);
    setVector(&v2, min.x, max.y, max.z);
    setTriangle(&tris[4], v2, v1, v0);

    setVector(&v0, min.x, max.y, max.z);
    setVector(&v1, max.x, max.y, max.z);
    setVector(&v2, max.x, min.y, max.z);
    setTriangle(&tris[5], v2, v1, v0);

    /* left */
    setVector(&v0, min.x, min.y, max.z);
    setVector(&v1, min.x, min.y, min.z);
    setVector(&v2, min.x, max.y, min.z);
    setTriangle(&tris[6], v2, v1, v0);

    setVector(&v0, min.x, max.y, min.z);
    setVector(&v1, min.x, max.y, max.z);
    setVector(&v2, min.x, min.y, max.z);
    setTriangle(&tris[7], v2, v1, v0);

    /* bottom */
    setVector(&v0, min.x, min.y, min.z);
    setVector(&v1, min.x, min.y, max.z);
    setVector(&v2, max.x, min.y, max.z);
    setTriangle(&tris[8], v2, v1, v0);

    setVector(&v0, max.x, min.y, max.z);
    setVector(&v1, max.x, min.y, min.z);
    setVector(&v2, min.x, min.y, min.z);
    setTriangle(&tris[9], v2, v1, v0);

    /* top */
    setVector(&v0, min.x, max.y, min.z);
    setVector(&v1, max.x, max.y, min.z);
    setVector(&v2, max.x, max.y, max.z);
    setTriangle(&tris[10], v2, v1, v0);

    setVector(&v0, max.x, max.y, max.z);
    setVector(&v1, min.x, max.y, max.z);
    setVector(&v2, min.x, max.y, min.z);
    setTriangle(&tris[11], v2, v1, v0);

    /* finally set the object */
    setObject(object, material, 12, tris);
}

void createCylinder(Object *object, Material material, float radius, float height, int resolution)
{
    /* resolution is linear sampling here, min can be 4 */
    int i;
    float t1, t2;
    Vector v0, v1, v2, v3, top, origin;
    Triangle *tris;

    /* 4 times the resolution, x2 because cylinder columns, x2 to cover both bases */
    tris = (Triangle *)malloc(sizeof(Triangle) * 4 * resolution);

    setVector(&top,    0.0f, height, 0.0f);
    setVector(&origin, 0.0f, 0.0f,   0.0f);

    for(i = 0; i < resolution; i++)
    {
        t1 = (float)i / (float)resolution;
        t1 = deg2rad(t1 * 360.0f);
        t2 = (float)(i + 1) / (float)resolution;
        t2 = deg2rad(t2 * 360.0f);

        setVector(&v0, radius * cos(t1), 0.0f, radius * sin(t1));
        setVector(&v1, radius * cos(t2), 0.0f, radius * sin(t2));
        setVector(&v2, v1.x, height, v1.z);
        setVector(&v3, v0.x, height, v0.z);

        /* Cylinder structure */
        setTriangle(&tris[4 * i + 0], v2, v1, v0);
        setTriangle(&tris[4 * i + 1], v0, v3, v2);

        /* Cylinder base */
        setTriangle(&tris[4 * i + 2], v1, origin, v0);

        /* Cylinder top */
        setTriangle(&tris[4 * i + 3], top, v2, v3);
    }

    /* finally set the object */
    setObject(object, material, 4 * resolution, tris);
}

/* Transform Object */

void transformObject(Object *object, Matrix M)
{
    int i;
    Triangle temp;

    //nth = omp_get_num_threads();

    #pragma omp parallel for
      for(i = 0; i < (*object).ntris; i++)
      {
          //printf(" Threads: %d\n", i);
          temp.v0 = matVecMult(M, ((*object).tri[i].v0));
          temp.v1 = matVecMult(M, ((*object).tri[i].v1));
          temp.v2 = matVecMult(M, ((*object).tri[i].v2));
          (*object).tri[i].v0 = temp.v0;
          (*object).tri[i].v1 = temp.v1;
          (*object).tri[i].v2 = temp.v2;
      }

}

#endif
