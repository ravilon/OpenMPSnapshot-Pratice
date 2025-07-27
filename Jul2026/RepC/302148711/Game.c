
#include "Game.h"

static inline uint8_t get_neighbors(unsigned int x, unsigned int y) {
    // Check all neighbor fields
    uint8_t alive = 0;

    // Check in current row
    {
        // Left
        if (x > 0) {
            if (world_get(x - 1, y) == ALIVE) {
                alive++;
            }
        }
        // Right
        if (x + 1 < WORLD_WIDTH) {
            if (world_get(x + 1, y) == ALIVE) {
                alive++;
            }
        }
    }

    // Check in upper row
    if (y > 0) {
        // Upper left
        if (x > 0) {
            if (world_get(x - 1, y - 1) == ALIVE) {
                alive++;
            }
        }
        // Upper
        if (world_get(x, y - 1) == ALIVE) {
            alive++;
        }
        // Upper right
        if (x + 1 < WORLD_WIDTH) {
            if (world_get(x + 1, y - 1) == ALIVE) {
                alive++;
            }
        }
    }

    // Check in lower row
    if (y + 1 < WORLD_HEIGHT) {
        // Lower left
        if (x > 0) {
            if (world_get(x - 1, y + 1) == ALIVE) {
                alive++;
            }
        }
        // Lower
        if (world_get(x, y + 1) == ALIVE) {
            alive++;
        }
        // Lower right
        if (x + 1 < WORLD_WIDTH) {
            if (world_get(x + 1, y + 1) == ALIVE) {
                alive++;
            }
        }
    }

    return alive;
}

void step() {
    unsigned long capacity = WORLD_WIDTH * WORLD_HEIGHT;

    // Create new world which will be filled in this function and
    // will then replace the old world.
    unsigned char *new_world = BitVector_init(capacity);

    // Check all fields in current world.
    // Always check 8 fields in one go, put them in a byte and
    // then write this whole byte to the new world.
    // => This can be parallelized without locks because only
    //    on thread is every writing to one byte of the BitVector.
    unsigned char byte = 0x00;
    #pragma omp parallel for private(byte)
    // TODO: Tweak with omp schedule() statement?
    for (unsigned int i = 0; i < capacity; i += 8) {
        // Byte which will contain the 8 fields set in this loop run
        byte = 0x00;

        // Check 8 fields in one run
        for (int k = 0; k < 8; k++) {
           unsigned int x = (i + k) % WORLD_WIDTH;
           unsigned int y = (i + k) / WORLD_WIDTH;

           // Check how many neighbors are alive
           int8_t alive = get_neighbors(x, y);

           // Decide what happens with field depending on
           // number of  neighbors who are alive
           if (alive > 3) {
               // Stay dead or die
           } else if (alive == 3) {
               // Stay alive or get born
               byte |= (ALIVE << k);
           } else if (alive == 2) {
               // Stay alive or stay dead
               byte |= (world_get(x, y) << k);
           } else if (alive < 2) {
               // Stay dead or die
           }
        }

        // Write
        BitVector_set8(new_world, i / 8, byte);
    }

    // Apply new world
    WORLD = new_world;
}

