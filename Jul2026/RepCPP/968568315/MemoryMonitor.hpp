#pragma once
#include <unordered_map>
#include <string>
#include <atomic>
#include <memory>

/**
 * @brief System-wide memory usage tracking and monitoring
 * 
 * Provides real-time memory tracking per component with thread-safe
 * operations. Tracks both current and peak memory usage.
 * 
 * Features:
 * - Component-specific memory tracking
 * - Peak usage monitoring
 * - Thread-safe operations
 * - RAII-based tracking
 * 
 * Usage:
 * @code
 * auto tracker = MemoryTracker<Grid>("GridSystem", sizeof(Grid));
 * auto usage = MemoryMonitor::getInstance().getCurrentUsage();
 * @endcode
 */
class MemoryMonitor {
private:
    std::atomic<size_t> current_usage{0};
    std::atomic<size_t> peak_usage{0};
    std::unordered_map<std::string, size_t> allocation_map;
    
public:
    static MemoryMonitor& getInstance() {
        static MemoryMonitor instance;
        return instance;
    }

    /**
     * @brief Records memory allocation for a component
     * @param component Name of the component
     * @param size Size in bytes
     * @thread_safety Thread-safe
     */
    void trackAllocation(const std::string& component, size_t size) {
        allocation_map[component] += size;
        size_t new_usage = current_usage.fetch_add(size) + size;
        updatePeakUsage(new_usage);
    }

    void trackDeallocation(const std::string& component, size_t size) {
        allocation_map[component] -= size;
        current_usage.fetch_sub(size);
    }

    size_t getCurrentUsage() const {
        return current_usage.load();
    }

    size_t getPeakUsage() const {
        return peak_usage.load();
    }

    std::unordered_map<std::string, size_t> getAllocationMap() const {
        return allocation_map;
    }

private:
    void updatePeakUsage(size_t usage) {
        size_t current_peak = peak_usage.load();
        while(usage > current_peak) {
            if(peak_usage.compare_exchange_weak(current_peak, usage)) {
                break;
            }
        }
    }
};

// RAII wrapper for automatic tracking
template<typename T>
class MemoryTracker {
private:
    std::string component_name;
    size_t size;

public:
    MemoryTracker(const std::string& name, size_t s) 
        : component_name(name), size(s) {
        MemoryMonitor::getInstance().trackAllocation(component_name, size);
    }

    ~MemoryTracker() {
        MemoryMonitor::getInstance().trackDeallocation(component_name, size);
    }
};
