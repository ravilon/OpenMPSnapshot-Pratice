#pragma once
#include <memory>
#include "IDrawStrategy.hpp"
#include "Strategy/DrawCrossStrategy.hpp"
#include "Strategy/DrawCrossOpenMPStrategy.hpp"
#include "Strategy/DrawCrossThreadStrategy.hpp"

/**
 * @class DrawStrategyFactory
 * @brief Factory class for creating different drawing strategy implementations
 */
class DrawStrategyFactory {
public:
    /**
     * @enum StrategyType
     * @brief Available drawing strategy types
     */
    enum class StrategyType {
        NONE,      ///< Single-threaded implementation
        OPENMP,    ///< OpenMP multi-threaded implementation
        THREAD     ///< POSIX threads implementation
    };

    /**
     * @brief Creates a drawing strategy instance
     * @param type Strategy type to create
     * @return Unique pointer to the strategy implementation
     * @throws std::invalid_argument for unknown strategy types
     */
    static std::unique_ptr<IDrawStrategy> create(StrategyType type) {
        switch(type) {
            case StrategyType::NONE:
                return std::make_unique<DrawCrossStrategy>();
            case StrategyType::OPENMP:
                return std::make_unique<DrawCrossOpenMPStrategy>();
            case StrategyType::THREAD:
                return std::make_unique<DrawCrossThreadStrategy>();
            default:
                throw std::invalid_argument("Unknown strategy type");
        }
    }
};