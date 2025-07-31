#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <omp.h>
#include "renderer.hpp"
using namespace std;

Renderer::Renderer(int w, int h) : width(w), height(h) {
    canvas = new Color[width * height];         // canvas = new PaddedColor[width * height];    to address False Sharing
    for(int i = 0; i < width * height; i++) {
        canvas[i] = Color(1.0f, 1.0f, 1.0f);
        // canvas[i].r = 1.0f;
        // canvas[i].g = 1.0f;
        // canvas[i].b = 1.0f;  to address False Sharing
    }
}

Renderer::~Renderer() {
    delete[] canvas;
}

// Checks if a pixel belongs to a circle
bool Renderer::isPixelInCircle(int x, int y, const Circle& circle) const {
    float dx = x - circle.x;
    float dy = y - circle.y;
    return (dx * dx + dy * dy) <= (circle.radius * circle.radius);
}

// Alpha blending determines new pixel color
Color Renderer::alphaBlending(const Color& source, const Color& dest, float alpha) const {
    return Color(
            source.r * alpha + dest.r * (1 - alpha),
            source.g * alpha + dest.g * (1 - alpha),
            source.b * alpha + dest.b * (1 - alpha)
            );
}

void Renderer::processPixel(int x, int y) {
    Color finalColor;       // Color finalColor(1.0f, 1.0f, 1.0f);      to address False Sharing

    for (const Circle& circle : circles) {
        if (isPixelInCircle(x, y, circle)) {
            finalColor = alphaBlending(circle.color, finalColor, circle.alpha);
        }
    }
    // canvas = 1D-array representing a 2D grid. In a 5x5 grid, pixel (2,1) lies at index 1 * 5 + 2 = 7
    canvas[y * width + x] = finalColor;
    // canvas[y * width + x].r = finalColor.r;
    // canvas[y * width + x].g = finalColor.g;
    // canvas[y * width + x].b = finalColor.b;      to address False Sharing
}

void Renderer::addCircle(const Circle& circle) {
    circles.push_back(circle);
}

void Renderer::saveToPPM(const string& filename) {
    ofstream file(filename, ios::binary);
    file << "P6\n" << width << " " << height << "\n255\n";

    for(int i = 0; i < width * height; i++) {
        unsigned char r = static_cast<unsigned char>(canvas[i].r * 255);
        unsigned char g = static_cast<unsigned char>(canvas[i].g * 255);
        unsigned char b = static_cast<unsigned char>(canvas[i].b * 255);
        file.write(reinterpret_cast<char*>(&r), 1);
        file.write(reinterpret_cast<char*>(&g), 1);
        file.write(reinterpret_cast<char*>(&b), 1);
    }
}

SequentialResult Renderer::renderSequential() {
    SequentialResult result;

    // 1. Sorting circles accordingly with their z coordinate
    auto startSort = chrono::high_resolution_clock::now();
    sort(circles.begin(), circles.end(),[](const Circle& c1, const Circle& c2) { return c1.z > c2.z; });
    auto endSort = chrono::high_resolution_clock::now();
    result.seqSortingTime = chrono::duration_cast<chrono::milliseconds>(endSort - startSort).count();

    // 2. Blending colors for all pixels, checking whether a circle belongs to a pixel
    auto startRender = chrono::high_resolution_clock::now();
    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            processPixel(x, y);
        }
    }
    auto endRender = chrono::high_resolution_clock::now();
    result.seqRenderTime = chrono::duration_cast<chrono::milliseconds>(endRender - startRender).count();

    result.seqExecutionTime = result.seqSortingTime + result.seqRenderTime;

    return result;
}

ParallelResult Renderer::renderParallel(int numThreads, int blockSize, float seqExecutionTime) {
    ParallelResult result;
    result.numThreads = numThreads;
    result.blockSize = blockSize;

    // 1. Sorting circles accordingly with their z coordinate
    auto startSort = chrono::high_resolution_clock::now();
    sort(circles.begin(), circles.end(),[](const Circle& c1, const Circle& c2) { return c1.z > c2.z; });
    auto endSort = chrono::high_resolution_clock::now();
    result.parSortingTime = chrono::duration_cast<chrono::milliseconds>(endSort - startSort).count();

    // 2. Blending colors for all pixels, checking whether a circle belongs to a pixel
    auto startRender = chrono::high_resolution_clock::now();
    #pragma omp parallel for num_threads(numThreads)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                processPixel(x, y);
            }
        }
    auto endRender = chrono::high_resolution_clock::now();
    result.parRenderTime = chrono::duration_cast<chrono::milliseconds>(endRender - startRender).count();

    result.parExecutionTime = result.parSortingTime + result.parRenderTime;
    result.speedup = seqExecutionTime/result.parExecutionTime;
    result.efficiency = result.speedup/result.numThreads;

    return result;
}
