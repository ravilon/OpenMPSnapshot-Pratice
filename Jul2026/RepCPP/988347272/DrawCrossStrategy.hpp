#pragma once
#include "IDrawStrategy.hpp"
#include <cmath>

/**
 * @class DrawCrossStrategy
 * @brief Single-threaded implementation of cross drawing strategy
 * 
 * Basic implementation using Bresenham's line algorithm for drawing
 */
class DrawCrossStrategy : public IDrawStrategy {
public:
    /**
     * @brief Constructor
     * @param color Initial drawing color (default: black)
     * @param thickness Initial line thickness (default: 1)
     */
    explicit DrawCrossStrategy(BMPFile::Pixel color = {0, 0, 0, 255}, 
                            unsigned int thickness = 1);

    void draw(BMPFile& image) override;
    std::string getName() const override;
    
    void setColor(const BMPFile::Pixel& color) override;
    BMPFile::Pixel getColor() const override;
    
    void setThickness(unsigned int thickness) override;
    unsigned int getThickness() const override;

private:
    BMPFile::Pixel color_;
    unsigned int thickness_;

    void drawLine(BMPFile& image, int x0, int y0, int x1, int y1);
    void drawThickPixel(BMPFile& image, int x, int y);
};