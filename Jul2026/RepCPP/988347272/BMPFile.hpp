/**
 * @file BMPFile.hpp
 * @brief Class for working with BMP images
 */

#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <fstream>

/**
 * @class BMPFile
 * @brief Class for loading, saving and processing BMP images
 * 
 * Supports 24-bit (BGR) and 32-bit (BGRA) pixel formats.
 * Provides basic image operations: read/write, pixel access, 
 * vertical flipping, conversion to black and white.
 */
class BMPFile {
public:
    /**
     * @enum PixelFormat
     * @brief Pixel storage format in the image
     */
    enum class PixelFormat {
        BGR24,  ///< 24-bit format (blue, green, red)
        BGRA32  ///< 32-bit format (blue, green, red, alpha channel)
    };

    #pragma pack(push, 1)
    /**
     * @struct BMPHeader
     * @brief BMP file header
     */
    struct BMPHeader {
        uint16_t signature = 0x4D42;  ///< "BM" signature
        uint32_t file_size = 0;       ///< File size in bytes
        uint16_t reserved1 = 0;       ///< Reserved
        uint16_t reserved2 = 0;       ///< Reserved
        uint32_t data_offset = 0;     ///< Offset to pixel data
    };

    /**
     * @struct DIBHeader
     * @brief BMP information header (DIB)
     */
    struct DIBHeader {
        uint32_t header_size = 40;     ///< Size of this header (40 bytes)
        int32_t width = 0;            ///< Image width in pixels
        int32_t height = 0;           ///< Image height in pixels (negative for top-down)
        uint16_t planes = 1;          ///< Number of planes (always 1)
        uint16_t bits_per_pixel = 24; ///< Bits per pixel (24 or 32)
        uint32_t compression = 0;     ///< Compression type (0 = no compression)
        uint32_t image_size = 0;      ///< Image data size
        int32_t x_pixels_per_meter = 0; ///< Horizontal resolution (pixels/meter)
        int32_t y_pixels_per_meter = 0; ///< Vertical resolution (pixels/meter)
        uint32_t colors_used = 0;     ///< Number of colors in palette
        uint32_t important_colors = 0; ///< Number of important colors
    };
    #pragma pack(pop)

    /**
     * @struct Pixel
     * @brief Structure representing a pixel
     */
    struct Pixel {
        uint8_t b = 0;  ///< Blue component
        uint8_t g = 0;  ///< Green component
        uint8_t r = 0;  ///< Red component
        uint8_t a = 255; ///< Alpha channel (transparency)

        Pixel() = default;
        
        /**
         * @brief Pixel constructor
         * @param red Red component (0-255)
         * @param green Green component (0-255)
         * @param blue Blue component (0-255)
         * @param alpha Alpha channel (0-255, default 255)
         */
        Pixel(uint8_t red, uint8_t green, uint8_t blue, uint8_t alpha = 255)
            : b(blue), g(green), r(red), a(alpha) {}

        /**
         * @brief Pixel comparison operator
         * @param other Other pixel to compare with
         * @return true if pixels are identical, false otherwise
         */
        bool operator==(const Pixel& other) const {
            return r == other.r && g == other.g && b == other.b;
        }
    };

    BMPFile() = default;
    ~BMPFile() = default;

    /**
     * @brief Loads BMP image from file
     * @param filename Path to the file
     * @return true if loading succeeded, false on error
     */
    bool load(const std::string& filename);
    
    /**
     * @brief Saves BMP image to file
     * @param filename Path to the file
     * @return true if saving succeeded, false on error
     */
    bool save(const std::string& filename) const;

    /**
     * @brief Gets image width
     * @return Width in pixels
     */
    int width() const { return dib_header_.width; }
    
    /**
     * @brief Gets image height
     * @return Height in pixels (always positive)
     */
    int height() const { return std::abs(dib_header_.height); }
    
    /**
     * @brief Checks if image is 32-bit format
     * @return true if 32-bit (with alpha channel), false if 24-bit
     */
    bool is32bit() const { return dib_header_.bits_per_pixel == 32; }

    /**
     * @brief Gets pixel by coordinates
     * @param x X coordinate (0..width-1)
     * @param y Y coordinate (0..height-1)
     * @return Pixel
     * @throw std::out_of_range if coordinates are out of bounds
     */
    Pixel getPixel(int x, int y) const;
    
    /**
     * @brief Sets pixel by coordinates
     * @param x X coordinate (0..width-1)
     * @param y Y coordinate (0..height-1)
     * @param pixel New pixel value
     * @throw std::out_of_range if coordinates are out of bounds
     */
    void setPixel(int x, int y, Pixel pixel);
    
    /**
     * @brief Flips image vertically
     */
    void flipVertically();
    
    /**
     * @brief Converts image to black and white
     */
    void convertToBlackAndWhite();

    /**
     * @brief Creates a new blank BMP image
     */
    void create(int width, int height, PixelFormat format, Pixel fill_color);

private:
    BMPHeader bmp_header_;          ///< BMP file header
    DIBHeader dib_header_;          ///< Information header
    std::vector<Pixel> pixels_;     ///< Image pixel array

    /**
     * @brief Reads headers from file
     * @param file File stream
     */
    void readHeaders(std::ifstream& file);
    
    /**
     * @brief Reads pixel data from file
     * @param file File stream
     */
    void readPixels(std::ifstream& file);
    
    /**
     * @brief Writes headers to file
     * @param file File stream
     */
    void writeHeaders(std::ofstream& file) const;
    
    /**
     * @brief Calculates row size with padding
     * @return Row size in bytes
     */
    size_t getRowSize() const;
    
    /**
     * @brief Calculates index in pixel array
     * @param x X coordinate
     * @param y Y coordinate
     * @return Index in pixels_ array
     */
    size_t index(int x, int y) const;
    
    /**
     * @brief Checks if coordinates are within image bounds
     * @param x X coordinate
     * @param y Y coordinate
     * @return true if coordinates are valid, false otherwise
     */
    bool inBounds(int x, int y) const;
    
    /**
     * @brief Adjusts row index according to image orientation
     * @param y Y coordinate
     * @return Row index in pixel array
     */
    int rowIndex(int y) const;
};