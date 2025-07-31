#include <chrono>
#include <cstdint>
#include <random>
#include <vector>

struct particle {
    float x, y, z;     // position
    float vx, vy, vz;  // velocity
    float ax, ay, az;  // acceleration
};

typedef std::vector<particle> particles;

void forces(particles &plist) {
    int n = plist.size();
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {  // We want to calculate the force on all particles
        plist[i].ax = plist[i].ay = plist[i].az = 0;  // start with zero acceleration
                                                      // #pragma omp parallel for
        for (int j = 0; j < n; ++j) {                 // Depends on all other particles
            if (i == j) continue;                     // Skip self interaction
            auto dx = plist[j].x - plist[i].x;
            auto dy = plist[j].y - plist[i].y;
            auto dz = plist[j].z - plist[i].z;
            auto r = sqrt(dx * dx + dy * dy + dz * dz);
            auto ir3 = 1 / (r * r * r);
            plist[i].ax += dx * ir3;
            plist[i].ay += dy * ir3;
            plist[i].az += dz * ir3;
        }
    }
}

// Initial conditions
void ic(particles &plist, int n) {
    std::random_device rd;   // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    plist.clear();     // Remove all particles
    plist.reserve(n);  // Make room for "n" particle
    for (auto i = 0; i < n; ++i) {
        particle p{dis(gen), dis(gen), dis(gen), 0, 0, 0, 0, 0, 0};
        plist.push_back(p);
    }
}

int main(int argc, char *argv[]) {
    int N;  // number of particles
    if (argc != 2) {
        std::printf("No Particles given, using default of 20'000\n");
        N = 20'000;
    } else {
        uint32_t input = std::atoi(argv[1]);
        if (input == 0) {
            std::printf("Not a number! Exiting\n");
            return 1;
        } else {
            N = input;
        }
    }
    auto starttime = std::chrono::high_resolution_clock::now();

    particles plist;  // vector of particles
    ic(plist, N);     // initialize starting position/velocity
    forces(plist);    // calculate the forces

    auto endtime = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(endtime - starttime);
    std::printf("Time taken in ms: %d\n", diff);

    return 0;
}
