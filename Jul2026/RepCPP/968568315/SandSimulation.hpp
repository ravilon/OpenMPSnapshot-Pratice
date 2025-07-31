#pragma once
#include "../spatial/grid_spatial_connector.hpp"
#include "../ui/GridVisualizer.hpp"
#include <memory>
#include <string>

class SandSimulation {
private:
    std::unique_ptr<Grid> grid;
    std::unique_ptr<SpatialHash> spatialHash;
    std::unique_ptr<GridSpatialConnector> connector;
    std::unique_ptr<GridOperations> gridOps;
    std::unique_ptr<GridVisualizer> visualizer;
    
    bool running = false;
    int windowWidth = 800;
    int windowHeight = 600;
    int cellSize = 5;
    
    // Tool state
    ParticleType currentParticleType = ParticleType::SAND;
    int brushSize = 3;
    
public:
    SandSimulation(int width, int height, int cellSize = 5) 
        : windowWidth(width)
        , windowHeight(height)
        , cellSize(cellSize)
    {
        // Calculate grid dimensions based on window size and cell size
        uint32_t gridWidth = windowWidth / cellSize;
        uint32_t gridHeight = windowHeight / cellSize;
        
        // Initialize components
        grid = std::make_unique<Grid>(gridWidth, gridHeight);
        spatialHash = std::make_unique<SpatialHash>();
        gridOps = std::make_unique<GridOperations>(*grid);
        connector = std::make_unique<GridSpatialConnector>(*grid, *spatialHash);
        visualizer = std::make_unique<GridVisualizer>(*grid, *gridOps, windowWidth, windowHeight, cellSize);
    }
    
    void run() {
        running = true;
        
        while (running) {
            handleEvents();
            update();
            render();
        }
    }
    
    void handleEvents();
    void update();
    void render();
    
    void addParticlesInRadius(int centerX, int centerY, int radius);
    void setParticleType(ParticleType type) { currentParticleType = type; }
    void setBrushSize(int size) { brushSize = size; }
};