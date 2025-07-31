#pragma once
#include "BMPFile.hpp"
#include "DrawStrategyFactory.hpp"
#include <memory>

/**
 * @class BMPProcessor
 * @brief Handles BMP image processing with various drawing strategies
 * 
 * Provides functionality to load, process, save and display BMP images
 * with configurable drawing strategies and parameters
 */
class BMPProcessor {
public:
    /**
     * @struct Config
     * @brief Configuration parameters for BMP processing
     */
    struct Config {
        std::string input_file;                            ///< Input BMP file path
        std::string output_file = "output.bmp";            ///< Output BMP file path
        std::pair<char, char> display_chars = {'#', ' '};  ///< Characters for console display (foreground, background)
        BMPFile::Pixel color = {0, 0, 0, 255};             ///< Drawing color (RGBA, default: opaque black)
        unsigned int thickness = 1;                        ///< Line thickness in pixels
        std::string strategy_name = "none";                ///< Drawing strategy name
        DrawStrategyFactory::StrategyType strategy_type = DrawStrategyFactory::StrategyType::NONE;  ///< Drawing strategy type

        /**
         * @brief Parse command line arguments into Config
         * @param argc Argument count
         * @param argv Argument values
         * @return Config structure with parsed parameters
         * @throws std::runtime_error on invalid arguments
         */
        static Config parse(int argc, char* argv[]);
        
        /**
         * @brief Display help message with usage instructions
         * @param program_name Name of the executable (argv[0])
         */
        static void printHelp(const std::string& program_name);
    };

    /**
     * @brief Construct a new BMPProcessor object
     * @param config Processing configuration
     * @param strategy Drawing strategy implementation
     */
    BMPProcessor(const Config& config, std::unique_ptr<IDrawStrategy> strategy);
    
    /**
     * @brief Process the BMP image (load, draw, save)
     * @return true if processing succeeded, false otherwise
     */
    bool process();
    
    /**
     * @brief Display the image in console using configured characters
     */
    void display() const;
    
private:
    Config config_;                                 ///< Processing configuration
    BMPFile bmp_;                                   ///< BMP image handler
    std::unique_ptr<IDrawStrategy> draw_strategy_;  ///< Drawing strategy implementation
};