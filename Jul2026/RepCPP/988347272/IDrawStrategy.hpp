#pragma once
#include "BMPFile.hpp"
#include <memory>

class IDrawStrategy {
public:
    virtual ~IDrawStrategy() = default;
    
    /**
     * @brief Main drawing method
     * @param image Reference to BMP image to draw on
     */
    virtual void draw(BMPFile& image) = 0;
    
    /**
     * @brief Gets the name of the drawing strategy
     * @return Name of the strategy
     */
    virtual std::string getName() const = 0;
    
    /**
     * @brief Sets the drawing color
     * @param color New color to use for drawing
     */
    virtual void setColor(const BMPFile::Pixel& color) = 0;
    
    /**
     * @brief Gets the current drawing color
     * @return Current drawing color
     */
    virtual BMPFile::Pixel getColor() const = 0;
    
    /**
     * @brief Sets the line thickness
     * @param thickness New thickness value (must be >= 1)
     */
    virtual void setThickness(unsigned int thickness) = 0;
    
    /**
     * @brief Gets the current line thickness
     * @return Current thickness value
     */
    virtual unsigned int getThickness() const = 0;
};