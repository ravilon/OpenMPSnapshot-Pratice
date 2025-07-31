#pragma once
#include <vector>
#include <array>
#include <cassert>
#include <memory>
#include "MemoryMonitor.hpp"

/**
 * @brief Generic memory pool for efficient object allocation
 * 
 * Provides chunk-based memory management with O(1) allocation/deallocation.
 * Reduces memory fragmentation through fixed-size chunks and maintains
 * a free list for quick reuse of freed memory.
 * 
 * Performance characteristics:
 * - Allocation: O(1)
 * - Deallocation: O(1)
 * - Memory overhead: 1 pointer per object + chunk management
 * 
 * Usage:
 * @code
 * MemoryPool<Particle> pool;
 * Particle* p = pool.allocate();
 * pool.deallocate(p);
 * @endcode
 */
template<typename T>
class MemoryPool {
private:
    static constexpr size_t CHUNK_SIZE = 1024;
    std::vector<std::array<T, CHUNK_SIZE>> chunks;
    std::vector<T*> free_list;
    std::unordered_set<T*> allocated_ptrs;  // Track active allocations
    std::unique_ptr<MemoryTracker<MemoryPool<T>>> memory_tracker;
    size_t total_allocated = 0;

public:
    MemoryPool() {
        memory_tracker = std::make_unique<MemoryTracker<MemoryPool<T>>>(
            "MemoryPool_" + std::string(typeid(T).name()),
            0
        );
    }

    /**
     * @brief Allocates a new object from the pool
     * @return Pointer to allocated object
     * @note Objects are not initialized
     */
    T* allocate() {
        if (free_list.empty()) {
            addChunk();
        }
        T* ptr = free_list.back();
        free_list.pop_back();
        allocated_ptrs.insert(ptr);  // Track allocation
        total_allocated++;
        return ptr;
    }

    void deallocate(T* ptr) {
        if (!ptr) {
            return;  // Invalid pointer, just return
        }
        
        // Check if this pointer is actually managed by this pool
        auto it = allocated_ptrs.find(ptr);
        if (it == allocated_ptrs.end()) {
            // This pointer wasn't allocated by this pool
            return;
        }
        
        allocated_ptrs.erase(it);  // Remove from tracking
        free_list.push_back(ptr);
        total_allocated--;
    }

    size_t getAllocatedCount() const {
        return total_allocated;
    }

    ~MemoryPool() {
        size_t total_size = chunks.size() * CHUNK_SIZE * sizeof(T);
        MemoryMonitor::getInstance().trackDeallocation(
            "MemoryPool_" + std::string(typeid(T).name()),
            total_size
        );
    }

private:
    void addChunk() {
        chunks.emplace_back();
        size_t chunk_size = sizeof(T) * CHUNK_SIZE;
        MemoryMonitor::getInstance().trackAllocation(
            "MemoryPool_" + std::string(typeid(T).name()),
            chunk_size
        );
        
        auto& new_chunk = chunks.back();
        for (size_t i = CHUNK_SIZE; i > 0; --i) {
            free_list.push_back(&new_chunk[i - 1]);
        }
    }

    bool ownsPointer(T* ptr) const {
        return allocated_ptrs.find(ptr) != allocated_ptrs.end();
    }
};
