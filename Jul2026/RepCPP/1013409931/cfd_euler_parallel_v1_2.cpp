#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <sstream>
#include <chrono>
#include <omp.h>


using namespace std;
// Functions to assist in checking correct results.

// Save vector to file
void saveVectorToFile(const std::vector<double>& vec, const std::string& filename) {
    std::ofstream outFile(filename);
    if (!outFile) {
        std::cerr << "Error opening file for writing.\n";
        return;
    }

    for (const auto& val : vec) {
        outFile << val << "\n";  // One value per line
    }

    outFile.close();
}

// Load vector from file
std::vector<double> loadVectorFromFile(const std::string& filename) {
    std::ifstream inFile(filename);
    if (!inFile) {
        std::cerr << "Error opening file for reading.\n";
        return {};
    }

    std::vector<double> vec;
    double val;
    while (inFile >> val) {
        vec.push_back(val);
    }

    inFile.close();
    return vec;
}

const double TOLERANCE = 1e-2; // For floating-point comparison
// Compare vectors with tolerance
bool vectorsEqual(const std::vector<double>& v1, const std::vector<double>& v2) {
    if (v1.size() != v2.size()) return false;

    for (size_t i = 0; i < v1.size(); ++i) {
        if (std::fabs(v1[i] - v2[i]) > TOLERANCE) {
            std::cout << "The difference at the unequal element (index=" << i << "): " << std::fabs(v1[i] - v2[i]) << std::endl;
            return false;
        }
    }

    return true;
}

// ------------------------------------------------------------
// Global parameters
// ------------------------------------------------------------
const double gamma_val = 1.4;   // Ratio of specific heats
const double CFL = 0.5;         // CFL number

// ------------------------------------------------------------
// Compute pressure from the conservative variables
// ------------------------------------------------------------
double pressure(double rho, double rhou, double rhov, double E) {
    double u = rhou / rho;
    double v = rhov / rho;
    double kinetic = 0.5 * rho * (u * u + v * v);
    return (gamma_val - 1.0) * (E - kinetic);
}

// ------------------------------------------------------------
// Compute flux in the x-direction
// ------------------------------------------------------------
void fluxX(double rho, double rhou, double rhov, double E, 
           double& frho, double& frhou, double& frhov, double& fE) {
    double u = rhou / rho;
    double p = pressure(rho, rhou, rhov, E);
    frho = rhou;
    frhou = rhou * u + p;
    frhov = rhov * u;
    fE = (E + p) * u;
}

// ------------------------------------------------------------
// Compute flux in the y-direction
// ------------------------------------------------------------
void fluxY(double rho, double rhou, double rhov, double E,
           double& frho, double& frhou, double& frhov, double& fE) {
    double v = rhov / rho;
    double p = pressure(rho, rhou, rhov, E);
    frho = rhov;
    frhou = rhou * v;
    frhov = rhov * v + p;
    fE = (E + p) * v;
}

// ------------------------------------------------------------
// Write flow field to VTK file
// ------------------------------------------------------------
void writeVTK(const vector<vector<double>>& rho, const vector<vector<double>>& rhou,
              const vector<vector<double>>& rhov, const vector<vector<double>>& E,
              const int Nx, const int Ny, const double dx, const double dy, const int step) {
    stringstream ss;
    ss << "flow_" << step << ".vtk";
    ofstream vtkFile(ss.str());
    
    // VTK header
    vtkFile << "# vtk DataFile Version 3.0\n";
    vtkFile << "Flow field at step " << step << "\n";
    vtkFile << "ASCII\n";
    vtkFile << "DATASET STRUCTURED_GRID\n";
    
    // Grid dimensions
    vtkFile << "DIMENSIONS " << Nx << " " << Ny << " 1\n";
    
    // Points (cell centers)
    vtkFile << "POINTS " << (Nx * Ny) << " float\n";
    for (int j = 1; j <= Ny; j++) {
        for (int i = 1; i <= Nx; i++) {
            double x = (i - 0.5) * dx;
            double y = (j - 0.5) * dy;
            vtkFile << x << " " << y << " 0\n";
        }
    }
    
    // Point data (velocities)
    vtkFile << "POINT_DATA " << (Nx * Ny) << "\n";
    vtkFile << "VECTORS velocity float\n";
    for (int j = 1; j <= Ny; j++) {
        for (int i = 1; i <= Nx; i++) {
            double u = rhou[i][j] / rho[i][j];
            double v = rhov[i][j] / rho[i][j];
            vtkFile << u << " " << v << " 0\n";
        }
    }
    
    vtkFile.close();
}

// ------------------------------------------------------------
// Main simulation routine
// ------------------------------------------------------------
int main(){
    // ----- Grid and domain parameters -----
    const int Nx = 200;         // Number of cells in x (excluding ghost cells)
    const int Ny = 100;         // Number of cells in y
    const double Lx = 2.0;      // Domain length in x
    const double Ly = 1.0;      // Domain length in y
    const double dx = Lx / Nx;
    const double dy = Ly / Ny;

    // Create 2D grids (with ghost cells: indices 0 to Nx+1 and 0 to Ny+1)
    vector<vector<double>> rho(Nx+2, vector<double>(Ny+2));
    vector<vector<double>> rhou(Nx+2, vector<double>(Ny+2));
    vector<vector<double>> rhov(Nx+2, vector<double>(Ny+2));
    vector<vector<double>> E(Nx+2, vector<double>(Ny+2));
    
    vector<vector<double>> rho_new(Nx+2, vector<double>(Ny+2));
    vector<vector<double>> rhou_new(Nx+2, vector<double>(Ny+2));
    vector<vector<double>> rhov_new(Nx+2, vector<double>(Ny+2));
    vector<vector<double>> E_new(Nx+2, vector<double>(Ny+2));
    
    // A mask to mark solid cells (inside the cylinder)
    vector<vector<bool>> solid(Nx+2, vector<bool>(Ny+2, false));

    // ----- Obstacle (cylinder) parameters -----
    const double cx = 0.5;      // Cylinder center x
    const double cy = 0.5;      // Cylinder center y
    const double radius = 0.1;  // Cylinder radius

    // ----- Free-stream initial conditions (inflow) -----
    const double rho0 = 1.0;
    const double u0 = 1.0;
    const double v0 = 0.0;
    const double p0 = 1.0;
    const double E0 = p0/(gamma_val - 1.0) + 0.5*rho0*(u0*u0 + v0*v0);

    // ----- Initialize grid and obstacle mask -----
    #pragma omp parallel for
    {
        for (int i = 0; i < Nx+2; i++){
            for (int j = 0; j < Ny+2; j++){
                // Compute cell center coordinates
                double x = (i - 0.5) * dx;
                double y = (j - 0.5) * dy;
                // Mark cell as solid if inside the cylinder
                if ((x - cx)*(x - cx) + (y - cy)*(y - cy) <= radius * radius) {
                    solid[i][j] = true;
                    // For a wall, we set zero velocity
                    rho[i][j] = rho0;
                    rhou[i][j] = 0.0;
                    rhov[i][j] = 0.0;
                    E[i][j] = p0/(gamma_val - 1.0);
                } else {
                    solid[i][j] = false;
                    rho[i][j] = rho0;
                    rhou[i][j] = rho0 * u0;
                    rhov[i][j] = rho0 * v0;
                    E[i][j] = E0;
                }
            }
        }
    }
    // ----- Determine time step from CFL condition -----
    double c0 = sqrt(gamma_val * p0 / rho0);
    double dt = CFL * min(dx, dy) / (fabs(u0) + c0)/2.0;

    // ----- Time stepping parameters -----
    const int nSteps = 2000;

    auto t1 = std::chrono::high_resolution_clock::now();
    std::vector<double> kinetics_printed((nSteps/50)); /*40 elements will be stored for later check.*/
    const int num_of_threads = 8;
    omp_set_num_threads(num_of_threads);

    // ----- Main time-stepping loop -----
    for (int n = 0; n < nSteps; n++){
        // --- Apply boundary conditions on ghost cells ---
        // Left boundary (inflow): fixed free-stream state
        #pragma omp parallel for
        {
            for (int j = 0; j < Ny+2; j++){
                rho[0][j] = rho0;
                rhou[0][j] = rho0*u0;
                rhov[0][j] = rho0*v0;
                E[0][j] = E0;
            }
        }
        // Right boundary (outflow): copy from the interior
        #pragma omp parallel for
        {
            for (int j = 0; j < Ny+2; j++){
                rho[Nx+1][j] = rho[Nx][j];
                rhou[Nx+1][j] = rhou[Nx][j];
                rhov[Nx+1][j] = rhov[Nx][j];
                E[Nx+1][j] = E[Nx][j];
            }
        }
        // Bottom boundary: reflective
        #pragma omp parallel for
        {
            for (int i = 0; i < Nx+2; i++){
                rho[i][0] = rho[i][1];
                rhou[i][0] = rhou[i][1];
                rhov[i][0] = -rhov[i][1];
                E[i][0] = E[i][1];
            }
        }
        // Top boundary: reflective
        #pragma omp parallel for
        {    
            for (int i = 0; i < Nx+2; i++){
                rho[i][Ny+1] = rho[i][Ny];
                rhou[i][Ny+1] = rhou[i][Ny];
                rhov[i][Ny+1] = -rhov[i][Ny];
                E[i][Ny+1] = E[i][Ny];
            }
        }
        /*Threads wait each other as if there was a barrier after each for cycle. */


        // --- Update interior cells using a Lax-Friedrichs scheme ---
        #pragma omp parallel for
        {
            for (int i = 1; i <= Nx; i++){
                for (int j = 1; j <= Ny; j++){
                    // If the cell is inside the solid obstacle, do not update it
                    if (solid[i][j]) {
                        rho_new[i][j] = rho[i][j];
                        rhou_new[i][j] = rhou[i][j];
                        rhov_new[i][j] = rhov[i][j];
                        E_new[i][j] = E[i][j];
                        continue;
                    }

                    // Compute a Lax averaging of the four neighboring cells
                    rho_new[i][j] = 0.25 * (rho[i+1][j] + rho[i-1][j] + rho[i][j+1] + rho[i][j-1]);
                    rhou_new[i][j] = 0.25 * (rhou[i+1][j] + rhou[i-1][j] + rhou[i][j+1] + rhou[i][j-1]);
                    rhov_new[i][j] = 0.25 * (rhov[i+1][j] + rhov[i-1][j] + rhov[i][j+1] + rhov[i][j-1]);
                    E_new[i][j] = 0.25 * (E[i+1][j] + E[i-1][j] + E[i][j+1] + E[i][j-1]);

                    // Compute fluxes
                    double fx_rho1, fx_rhou1, fx_rhov1, fx_E1;
                    double fx_rho2, fx_rhou2, fx_rhov2, fx_E2;
                    double fy_rho1, fy_rhou1, fy_rhov1, fy_E1;
                    double fy_rho2, fy_rhou2, fy_rhov2, fy_E2;
                    
                    /*Variables declared inside a parallel block are unique to each thread. This is ok.*/

                    fluxX(rho[i+1][j], rhou[i+1][j], rhov[i+1][j], E[i+1][j],
                        fx_rho1, fx_rhou1, fx_rhov1, fx_E1);
                    fluxX(rho[i-1][j], rhou[i-1][j], rhov[i-1][j], E[i-1][j],
                        fx_rho2, fx_rhou2, fx_rhov2, fx_E2);
                    fluxY(rho[i][j+1], rhou[i][j+1], rhov[i][j+1], E[i][j+1],
                        fy_rho1, fy_rhou1, fy_rhov1, fy_E1);
                    fluxY(rho[i][j-1], rhou[i][j-1], rhov[i][j-1], E[i][j-1],
                        fy_rho2, fy_rhou2, fy_rhov2, fy_E2);

                    // Apply flux differences
                    double dtdx = dt / (2 * dx);
                    double dtdy = dt / (2 * dy);
                    
                    rho_new[i][j] -= dtdx * (fx_rho1 - fx_rho2) + dtdy * (fy_rho1 - fy_rho2);
                    rhou_new[i][j] -= dtdx * (fx_rhou1 - fx_rhou2) + dtdy * (fy_rhou1 - fy_rhou2);
                    rhov_new[i][j] -= dtdx * (fx_rhov1 - fx_rhov2) + dtdy * (fy_rhov1 - fy_rhov2);
                    E_new[i][j] -= dtdx * (fx_E1 - fx_E2) + dtdy * (fy_E1 - fy_E2);
                }
            }
        }
        // Copy updated values back
        #pragma omp parallel for
        {
            for (int i = 1; i <= Nx; i++){
                for (int j = 1; j <= Ny; j++){
                    rho[i][j] = rho_new[i][j];
                    rhou[i][j] = rhou_new[i][j];
                    rhov[i][j] = rhov_new[i][j];
                    E[i][j] = E_new[i][j];
                }
            }
        }
        // Calculate total kinetic energy
        double total_kinetic = 0.0;
        #pragma omp parallel for reduction(+:total_kinetic)
        {
            for (int i = 1; i <= Nx; i++) {
                for (int j = 1; j <= Ny; j++) {
                    double u = rhou[i][j] / rho[i][j];
                    double v = rhov[i][j] / rho[i][j];
                    total_kinetic += 0.5 * rho[i][j] * (u * u + v * v);
                }
            }
        }

        // Optional: output progress and write VTK file every 50 time steps
        if (n % 50 == 0) {
            cout << "Step " << n << " completed, total kinetic energy: " << total_kinetic << endl;
            writeVTK(rho, rhou, rhov, E, Nx, Ny, dx, dy, n);
            kinetics_printed[(n/50)] = total_kinetic;
            
        }
        
    }
    /*
    saveVectorToFile(kinetics_printed,"kinetics_results_saved.txt");
    for (double i: kinetics_printed){
        std::cout << i << std::endl;
    }
    */
    auto t2 = std::chrono::high_resolution_clock::now();
    std::vector<double> correct_result = loadVectorFromFile("kinetics_results_saved.txt");
    bool res_correct_bool = vectorsEqual(correct_result, kinetics_printed);
    if (res_correct_bool == true){
        std::cout << "The results are correct." << std::endl;
    }
    else{
        std::cout << "The results are NOT correct." << std::endl;
    }
        
    std::cout << "The process took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count()
              << " milliseconds. \n";
    return 0;
}

