#include <mpi.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <float.h>
#include "utils.h"
#include "scene.h"
#include <stdio.h>
#include <time.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define IMAGE_WIDTH 800
#define IMAGE_HEIGHT 600
#define numSamples 20

typedef struct HitPayload {
    float hitDistance;
    float WorldPosition[3];
    float WorldNormal[3];

    int ObjectIndex;
} s_hitPayload;

typedef struct Ray {
    float origin[3];
    float direction[3];
} s_ray;

s_hitPayload TraceRay(s_ray ray);
s_hitPayload ClosestHit(s_ray ray, int objectIndex, float hitDistance);
s_hitPayload Miss(s_ray ray);
int PerPixel(float x, float y);

s_hitPayload ClosestHit(s_ray ray, int objectIndex, float hitDistance){

    s_hitPayload hitPayload = {
        .hitDistance = hitDistance,
        .ObjectIndex = objectIndex
    };

    s_sphere *closestSphere = &scene.spheres[objectIndex];
    float origin[3];
    float lightDir[] = {-1, -1, -1};
    float normal[3];
    int red, green, blue;

    for (int i = 0; i < 3; i++) {
        origin[i] = ray.origin[i] - closestSphere->position[i];
    }

    for (int i = 0; i < 3; i++) {
        hitPayload.WorldPosition[i] = origin[i] + ray.direction[i] * hitDistance;
    }

    for (int i = 0; i < 3; i++) {
        hitPayload.WorldNormal[i] = hitPayload.WorldPosition[i];
    }
    normalize(hitPayload.WorldNormal, 3);

    for (int i = 0; i < 3; i++) {
        hitPayload.WorldPosition[i] += closestSphere->position[i];
    }

    return hitPayload;
}

s_hitPayload Miss(s_ray ray){
    s_hitPayload HitPayload;
    HitPayload.hitDistance = -1;

    return HitPayload;
}

int PerPixel(float x, float y){
    s_ray ray = {
        .origin = {0, 0.01, 0.1},
        .direction = {x, -y, -1.0}
    };
    
    normalize(ray.direction, 3);

    int colorR = 0;
    int colorG = 0;
    int colorB = 0;

    int skyColorR = 153; 
    int skyColorG = 178;
    int skyColorB = 230;

    float multiplier = 1.0f;

    float lightDir[] = {-1, -1, -1};
    normalize(lightDir, 3);
    lightDir[0] = -lightDir[0];
    lightDir[1] = -lightDir[1];
    lightDir[2] = -lightDir[2];

    int bounces = 5;
    for (int i = 0; i < bounces; i++){
        s_hitPayload HitPayload = TraceRay(ray);

        if (HitPayload.hitDistance < 0){
            colorR += skyColorR * multiplier;
            colorG += skyColorG * multiplier;
            colorB += skyColorB * multiplier;
            break;
        }

        float lightIntensity = fmaxf(dotProduct(HitPayload.WorldNormal, lightDir), 0.0f);

        s_sphere *Sphere = &scene.spheres[HitPayload.ObjectIndex];
        int red = Sphere->material.color[0];
        int green = Sphere->material.color[1];
        int blue = Sphere->material.color[2];

        colorR += (int)(red * lightIntensity * multiplier);
        colorG += (int)(green * lightIntensity * multiplier);
        colorB += (int)(blue * lightIntensity * multiplier);

        multiplier *= 0.5f;

        for (int i = 0; i < 3; i++){
            ray.origin[i] = HitPayload.WorldPosition[i] + HitPayload.WorldNormal[i] * 0.001f;
        }

        float reflectedDirection[3];
        float randomVec3[3];
        randomVector3(randomVec3, -0.5, 0.5);
        scalarMultiply(randomVec3, Sphere->material.roughness);
        sum(randomVec3, HitPayload.WorldNormal, HitPayload.WorldNormal);

        reflect(ray.direction, HitPayload.WorldNormal, reflectedDirection);

        for (int i = 0; i < 3; i++){
            ray.direction[i] = reflectedDirection[i];
        }
    }

    colorR = fminf(fmaxf(colorR, 0), 255);
    colorG = fminf(fmaxf(colorG, 0), 255);
    colorB = fminf(fmaxf(colorB, 0), 255);

    return (colorR << 16) | (colorG << 8) | colorB;
}


s_hitPayload TraceRay(s_ray ray) {
    float origin[3];
    int closestSphere = -1;
    float hitDistance = FLT_MAX;

    for (int i = 0; i < scene.numSpheres; i++){
        s_sphere sphere = scene.spheres[i];

        origin[0] = ray.origin[0] - sphere.position[0];
        origin[1] = ray.origin[1] - sphere.position[1];
        origin[2] = ray.origin[2] - sphere.position[2];

        float a = dotProduct(ray.direction, ray.direction);
        float b = 2 * dotProduct(ray.direction, origin);
        float c = dotProduct(origin, origin) - pow(sphere.radius, 2);

        float discriminant = pow(b, 2) - 4 * a * c;


        if (discriminant < 0) {
            continue;
        }

        float closestT = (-b - sqrt(discriminant)) / (2 * a);
        if (closestT > 0 && closestT < hitDistance){
            hitDistance = closestT;
            closestSphere = (int)i;
        }
    }

    if (closestSphere < 0){
        return Miss(ray);
    }

    return ClosestHit(ray, closestSphere, hitDistance);
}

void saveImage(int *pixels, int width, int height, char *filename) {
    unsigned char *imageData = (unsigned char*)malloc(width * height * 3);
    if (!imageData) {
        printf("Erro ao alocar memória para a imagem!\n");
        return;
    }

    // Converter de int (RGB compacto) para buffer de bytes RGB
    for (int i = 0; i < width * height; i++) {
        imageData[i * 3 + 0] = (pixels[i] >> 16) & 0xFF; // Red
        imageData[i * 3 + 1] = (pixels[i] >> 8) & 0xFF;  // Green
        imageData[i * 3 + 2] = pixels[i] & 0xFF;         // Blue
    }

    // Salvar como PNG
    if (stbi_write_png(filename, width, height, 3, imageData, width * 3)) {
        printf("Imagem salva como %s\n", filename);
    } else {
        printf("Erro ao salvar a imagem!\n");
    }

    free(imageData);
}

int sequentialRayTracing() {
    float start, end;
    start = clock(); // Using clock() for timing in sequential execution

    int* image = malloc(IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(int));
    if (image == NULL) {
        fprintf(stderr, "Erro ao alocar memória para a imagem.\n");
        return -1;
    }

    printf("Executando ray tracing sequencial...\n");

    for (int y = 0; y < IMAGE_HEIGHT; y++) {
        for (int x = 0; x < IMAGE_WIDTH; x++) {
            int colorAcumulated[3] = {0, 0, 0};

            for (int i = 0; i < numSamples; i++) {
                float jitterX = ((float)rand() / RAND_MAX - 0.5f) / IMAGE_WIDTH;
                float jitterY = ((float)rand() / RAND_MAX - 0.5f) / IMAGE_HEIGHT;
                float normX = (2.0f * x) / IMAGE_WIDTH - 1.0f + jitterX;
                float normY = (2.0f * y) / IMAGE_HEIGHT - 1.0f + jitterY;

                int color = PerPixel(normX, normY);
                colorAcumulated[0] += (color >> 16) & 0xFF;
                colorAcumulated[1] += (color >> 8) & 0xFF;
                colorAcumulated[2] += color & 0xFF;
            }

            // Calculando a média das amostras
            int r = colorAcumulated[0] / numSamples;
            int g = colorAcumulated[1] / numSamples;
            int b = colorAcumulated[2] / numSamples;

            image[y * IMAGE_WIDTH + x] = (r << 16) | (g << 8) | b;
        }
    }

    saveImage(image, IMAGE_WIDTH, IMAGE_HEIGHT, "output_sequential.png");
    free(image);

    end = clock();
    printf("Tempo total sequencial: %f segundos\n", (end - start) / CLOCKS_PER_SEC);

    return 0;
}

int parallelRayTracing(int rank, int size) {
    double start, end;
    start = MPI_Wtime();

    int numThreads = 4;                 // Número de threads OpenMP por processo
    omp_set_num_threads(numThreads);    // Define número de threads

    int rowsPerProcess = IMAGE_HEIGHT / size;
    int startRow = rank * rowsPerProcess;
    int endRow = (rank == size - 1) ? IMAGE_HEIGHT : startRow + rowsPerProcess;

    int* localPixels = malloc(rowsPerProcess * IMAGE_WIDTH * sizeof(int));

    printf("Processo %d: calculando linhas %d a %d com %d threads...\n", rank, startRow, endRow - 1, numThreads);

    // Paralelizando o loop com OpenMP
    #pragma omp parallel for
    for (int y = startRow; y < endRow; y++) {
        for (int x = 0; x < IMAGE_WIDTH; x++) {
            int colorAcumulated[3] = {0, 0, 0};

            for (int i = 0; i < numSamples; i++) {
                float jitterX = ((float)rand() / RAND_MAX - 0.5f) / IMAGE_WIDTH;  // Pequeno deslocamento aleatório
                float jitterY = ((float)rand() / RAND_MAX - 0.5f) / IMAGE_HEIGHT;
                float normX = (2.0f * x) / IMAGE_WIDTH - 1.0f + jitterX;
                float normY = (2.0f * y) / IMAGE_HEIGHT - 1.0f + jitterY;

                int color = PerPixel(normX, normY);
                colorAcumulated[0] += (color >> 16) & 0xFF;
                colorAcumulated[1] += (color >> 8) & 0xFF;
                colorAcumulated[2] += color & 0xFF;
            }

            // Calculando a média das amostras
            int r = colorAcumulated[0] / numSamples;
            int g = colorAcumulated[1] / numSamples;
            int b = colorAcumulated[2] / numSamples;

            localPixels[(y - startRow) * IMAGE_WIDTH + x] = (r << 16) | (g << 8) | b;
        }
    }

    int* finalImage = NULL;
    if (rank == 0) {
        finalImage = malloc(IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(int));
    }

    MPI_Gather(localPixels,                 // Dados locais
        rowsPerProcess * IMAGE_WIDTH,       // Número de elementos a enviar
        MPI_INT,                            // Tipo de dado
        finalImage,                         // Dados de destino
        rowsPerProcess * IMAGE_WIDTH,       // Número de elementos a receber 
        MPI_INT,                            // Tipo de dado
        0,                                  // Rank do processo raiz                              
        MPI_COMM_WORLD);                    // Comunicador

    if (rank == 0) {
        printf("Processo 0: todos os dados recebidos. Salvando imagem...\n");
        saveImage(finalImage, IMAGE_WIDTH, IMAGE_HEIGHT, "output_parallel.png");
        free(finalImage);

        end = MPI_Wtime();
        printf("Tempo total paralelo: %f segundos\n", end - start);
    }

    free(localPixels);
    return 0;
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    parallelRayTracing(rank, size);

    if (rank == 0)
    {
        printf("--------------------------------------\n");
        sequentialRayTracing();
    }

    MPI_Finalize();
    return 0;
}