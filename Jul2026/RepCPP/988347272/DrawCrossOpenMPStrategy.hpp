#pragma once
#include "IDrawStrategy.hpp"
#include <cmath>
#include <omp.h>

/**
 * @class DrawCrossOpenMPStrategy
 * @brief OpenMP-optimized implementation of cross drawing strategy
 * 
 * Uses OpenMP directives for parallel pixel drawing operations
 */
class DrawCrossOpenMPStrategy : public IDrawStrategy {
public:
    /**
     * @brief Constructor
     * @param color Initial drawing color (default: black)
     * @param thickness Initial line thickness (default: 1)
     */
    explicit DrawCrossOpenMPStrategy(BMPFile::Pixel color = {0, 0, 0, 255}, 
                             unsigned int thickness = 1)
        : color_(color), thickness_(thickness) {}

    void draw(BMPFile& image) override;
    std::string getName() const override { return "Cross Drawing Strategy (OpenMP)"; }
    
    void setColor(const BMPFile::Pixel& color) override { color_ = color; }
    BMPFile::Pixel getColor() const override { return color_; }
    
    void setThickness(unsigned int thickness) override { 
        thickness_ = std::max(1u, thickness); 
    }
    unsigned int getThickness() const override { return thickness_; }

private:
    BMPFile::Pixel color_;
    unsigned int thickness_;

    void drawLine(BMPFile& image, int x0, int y0, int x1, int y1);
    void drawThickPixel(BMPFile& image, int x, int y);
};