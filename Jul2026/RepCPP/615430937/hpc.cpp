/*
    N-Body problem solution using openMP.
    Based on code provided by: Guido Giuntoli
    link: https://www.linkedin.com/pulse/2-optimizing-c-n-body-problem-openmp-guido-giuntoli/
*/

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <random>
#include <sstream>
#include <fstream>
#include <iomanip>

using namespace std;

// Define constants for sphere radius
constexpr double MIN_RADIUS = 1.0e1;
constexpr double MAX_RADIUS = 1.0e2;

std::uniform_real_distribution<double> distribution(MIN_RADIUS, MAX_RADIUS);
std::default_random_engine generator;

class Particle
{
public:
    double pos[3], vel[3];

    Particle()
    {
        // Initialize particle position
        // place the particles in a rectangle
        pos[0] = 0; // distribution(generator);
        pos[1] = 0; // distribution(generator);
        pos[2] = 0; // distribution(generator);

        // Initialize the particle velocities with random values
        vel[0] = distribution(generator); // vel[0] = 0;
        vel[1] = distribution(generator); // vel[1] = 0;
        vel[2] = distribution(generator); // vel[2] = 0;

        // Ensure the particle is placed inside a spherical region
        makeItASphere(pos[0], pos[1], pos[2]);
    }

    // Method to move the particle's position inside a spherical region
    void makeItASphere(double &x, double &y, double &z)
    {
        // Calculate the distance from the origin to the particle's position
        double r = sqrt(x * x + y * y + z * z);

        // Check if the particle is outside the sphere
        if (r > MAX_RADIUS)
        {
            // Scale the position coordinates down to the surface of the sphere
            x = x * MAX_RADIUS / r;
            y = y * MAX_RADIUS / r;
            z = z * MAX_RADIUS / r;
        }
        else if (r < MIN_RADIUS)
        {
            // Resample a new random position if the particle is inside the sphere
            x = distribution(generator);
            y = distribution(generator);
            z = distribution(generator);

            // recursively call the function to check again
            makeItASphere(x, y, z);
        }
    }
};

class Problem
{

public:
    Problem(double mass, double dt, unsigned numParticles) : mMass(mass),
                                                             mInverseMass(1.0 / mass),
                                                             mDt(dt),
                                                             mNumParticles(numParticles),
                                                             mParticles(new Particle[numParticles])
    {
    }

    ~Problem()
    {
        delete[] mParticles;
    }

    // Method to compute the forces and update the particle positions and velocities
    void integrate();

    // Getter method to access the particles
    const Particle *getParticles() const { return mParticles; }

private:
    const double mG = 6.6743e-11;
    const double mMass;
    const double mInverseMass;
    const double mDt;
    const unsigned mNumParticles;
    Particle *const mParticles;
};

// Function to compute the forces and update the particle positions and velocities
void Problem::integrate()
{

    // Calculate the constant factor for gravitational force
    const double Const = mG * mMass * mMass;

// Compute forces acting on each particle using parallel threads
#pragma omp parallel for
    for (int pi = 0; pi < mNumParticles; pi++)
    {
        // Initialize the force vector for particle pi
        double force[3] = {};

        // Calculate the total force acting on particle pi due to all other particles
        for (int pj = 0; pj < mNumParticles; pj++)
        {
            // Ignore the force contribution of the particle itself
            if (pj != pi)
            {
                // Compute the distance between the two particles
                const double dij[3] = {
                    mParticles[pj].pos[0] - mParticles[pi].pos[0],
                    mParticles[pj].pos[1] - mParticles[pi].pos[1],
                    mParticles[pj].pos[2] - mParticles[pi].pos[2]};

                // Compute the squared distance between particles pi and pj
                const double dist2 = dij[0] * dij[0] +
                                     dij[1] * dij[1] +
                                     dij[2] * dij[2];

                // Calculate the force scaling factor for gravitational force
                const double ConstDist2 = Const / dist2;
                const double idist = 1 / sqrt(dist2);

                // Compute the gravitational force vector between particles pi and pj
                // F = C * m * m / ||x2 - x1||^2 * (x2 - x1) / ||x2 - x1||
                force[0] += ConstDist2 * dij[0] * idist;
                force[1] += ConstDist2 * dij[1] * idist;
                force[2] += ConstDist2 * dij[2] * idist;
            }
        }

        // Update the velocity of particle pi based on the computed force and time step
        // dv / dt = a = F / m
        mParticles[pi].vel[0] += force[0] * mInverseMass * mDt;
        mParticles[pi].vel[1] += force[1] * mInverseMass * mDt;
        mParticles[pi].vel[2] += force[2] * mInverseMass * mDt;
    }

// Update the positions of the particles after the forces have been computed
// This update should be done after computing forces to avoid race conditions
#pragma omp parallel for
    for (int pi = 0; pi < mNumParticles; pi++)
    {
        // Update the position of particle pi based on its velocity and time step
        // dx / dt = v
        mParticles[pi].pos[0] += mParticles[pi].vel[0] * mDt;
        mParticles[pi].pos[1] += mParticles[pi].vel[1] * mDt;
        mParticles[pi].pos[2] += mParticles[pi].vel[2] * mDt;
    }
}
void writeVTKFile(const std::string &filename, const Particle *particles, unsigned numParticles)
{
    std::ofstream vtkFile(filename);

    if (!vtkFile)
    {
        std::cerr << "Error: Could not open VTK file for writing: " << filename << std::endl;
        return;
    }

    vtkFile << "# vtk DataFile Version 4.0\n";
    vtkFile << "Particle positions\n";
    vtkFile << "ASCII\n";
    vtkFile << "DATASET POLYDATA\n";
    vtkFile << "POINTS " << numParticles << " float\n";

    for (unsigned i = 0; i < numParticles; ++i)
    {
        vtkFile << std::setprecision(10) << particles[i].pos[0] << " "
                << particles[i].pos[1] << " "
                << particles[i].pos[2] << "\n";
    }

    // Define vertices for the points
    vtkFile << "VERTICES " << numParticles << " " << numParticles * 2 << "\n";
    for (unsigned i = 0; i < numParticles; ++i)
    {
        vtkFile << "1 " << i << "\n";
    }

    vtkFile.close();
}

int main()
{
    int a = omp_get_max_threads();
    printf("%d", a);
    omp_set_num_threads(32);
    const int nTimeSteps = 500;
    const double Mass = 1e12;
    const double dt = 1e-2;
    const unsigned numParticles = 10000;
    Problem problem(Mass, dt, numParticles);

    double start_time = omp_get_wtime();
    // Run the simulation for the specified number of time steps
    for (int ts = 0; ts < nTimeSteps; ts++)
    {

        // Write VTK file for the current time step
        // std::stringstream vtkFilenameStream;
        // vtkFilenameStream << "particles_" << ts << ".vtk";
        // std::string vtkFilename = vtkFilenameStream.str();
        // writeVTKFile(vtkFilename, problem.getParticles(), numParticles);

        // Integrate the particles over the current time step
        problem.integrate();

        // cout << "Time step: " << ts << endl;
        // for (int i = 0; i < numParticles; ++i)
        //{
        // cout << "Particle " << i << ": "
        //<< "x: " << problem.getParticles()[i].pos[0] << " "
        //<< "y: " << problem.getParticles()[i].pos[1] << " "
        //<< "z: " << problem.getParticles()[i].pos[2] << endl;
        //}
    }
    double time = omp_get_wtime() - start_time;
    std::cout << "Elapsed time: " << time << endl;
    return 0;
}