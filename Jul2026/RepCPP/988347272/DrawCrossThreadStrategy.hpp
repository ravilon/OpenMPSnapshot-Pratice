#pragma once
#include "IDrawStrategy.hpp"
#include <cmath>
#include <thread>
#include <vector>
#include <mutex>

/**
 * @class DrawCrossThreadStrategy
 * @brief Thread-based implementation of cross drawing strategy
 * 
 * Uses std::thread for parallel pixel drawing with workload balancing
 */
class DrawCrossThreadStrategy : public IDrawStrategy {
public:
    /**
     * @brief Constructor
     * @param color Initial drawing color (default: black)
     * @param thickness Initial line thickness (default: 1)
     */
    explicit DrawCrossThreadStrategy(BMPFile::Pixel color = {0, 0, 0, 255},
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
    mutable std::mutex mutex_;

    void drawLine(BMPFile& image, int x0, int y0, int x1, int y1);
    void drawThickPixel(BMPFile& image, int x, int y);
    void drawThickPixelArea(BMPFile& image, int x, int y);
};