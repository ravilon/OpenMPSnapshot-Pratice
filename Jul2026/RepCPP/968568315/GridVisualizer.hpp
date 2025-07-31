#pragma once
#include "../grid/GridOperations.hpp"
#include <SDL2/SDL.h>
#include <memory>
#include <string>

class GridVisualizer {
private:
    SDL_Window* window;
    SDL_Renderer* renderer;
    Grid& grid;
    GridOperations& gridOps;
    int cellSize;
    bool running;

public:
    GridVisualizer(Grid& g, GridOperations& ops, int windowWidth, int windowHeight, int cellSize = 5)
        : grid(g), gridOps(ops), cellSize(cellSize), running(false) {
        
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            throw std::runtime_error("SDL could not initialize! SDL_Error: " + std::string(SDL_GetError()));
        }
        
        window = SDL_CreateWindow("Sand Simulation", 
                                 SDL_WINDOWPOS_UNDEFINED, 
                                 SDL_WINDOWPOS_UNDEFINED, 
                                 windowWidth, 
                                 windowHeight, 
                                 SDL_WINDOW_SHOWN);
        
        if (!window) {
            throw std::runtime_error("Window could not be created! SDL_Error: " + std::string(SDL_GetError()));
        }
        
        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
        
        if (!renderer) {
            throw std::runtime_error("Renderer could not be created! SDL_Error: " + std::string(SDL_GetError()));
        }
    }
    
    ~GridVisualizer() {
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
    }
    
    void render();
    void handleEvents();
    void run();
};