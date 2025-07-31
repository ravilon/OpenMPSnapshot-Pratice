#pragma once

#include <backend/backend_interface.h>
#include <iomanip>

namespace sim {

/**
 * @class molecular_dynamics
 * @tparam T
 */
template<typename T, template<typename> class backend> class molecular_dynamics {
public:
    /**
     * Constructor, initializes the system.
     * @param particules
     * @param config
     */
    EXPORT molecular_dynamics(const std::vector<coordinate<T>>& particules, configuration<T> config, backend<T>&& be = {});

    EXPORT molecular_dynamics(const std::vector<coordinate<T>>& particules, const std::vector<coordinate<T>>& vitesses, configuration<T> config, backend<T>&& be = {});

    /**
     *
     */
    EXPORT void run_iter();

private:
    /**
     * Updates the kinectic energy and temperature based on current momentums.
     * @tparam T
     */
    void recompute_kinetic_energy_and_temp() noexcept;

    [[nodiscard]] inline size_t degrees_of_freedom() const noexcept { return 3 * backend_.get_particles_count() - 3; }

    /**
     * Updates the momentums to match the desierd kinetic temperature.
     * @tparam T
     * @param desired_temperature
     */
    void fixup_temperature(T desired_temperature) noexcept;

    /**
     * Applies the Berendsen thermostate on the current system using the current kinetic temperature.
     */
    void try_to_apply_berendsen_thermostate() noexcept;

private:
    [[no_unique_address]] const configuration<T> configuration_;   //
    [[no_unique_address]] size_t simulation_idx_;                  //
    [[no_unique_address]] pdb_writer writer_;                      //
    [[no_unique_address]] backend<T> backend_;                     //
    [[no_unique_address]] T kinetic_temperature_{};                //
    [[no_unique_address]] T kinetic_energy_{};                     //


    /* Metrics */
private:
    // Mutable variables are the variables that are easily and often re-computed based on the simulation
    // they should be updated as often as possible, but don't really affect the simulation at all
    [[no_unique_address]] mutable coordinate<T> forces_sum_{};     //
    [[no_unique_address]] mutable T lennard_jones_energy_{};       //
    [[no_unique_address]] mutable double avg_iter_duration_ = 1;   //
    [[no_unique_address]] mutable double total_energy_ = 0;        //
    [[no_unique_address]] mutable double avg_delta_energy_ = 0;    //


    /* Not very important stuff */
    EXPORT void update_display_metrics() const noexcept;

    EXPORT T compute_barycenter_speed_norm() const;

public:
    /**
     *
     * @param os
     * @param state
     * @return
     */
    EXPORT friend std::ostream& operator<<(std::ostream& os, const molecular_dynamics& state) {
        state.update_display_metrics();
        os << std::setprecision(10) << "[" << state.simulation_idx_ << "] "                                                //
           << "E_tot: " << std::setw(13) << std::setfill(' ') << state.total_energy_                                       //
           << ", Temp: " << std::setw(13) << std::setfill(' ') << state.kinetic_temperature_                               //
           << ", E_c: " << std::setw(13) << std::setfill(' ') << state.kinetic_energy_                                     //
           << ", E_pot: " << std::setw(13) << std::setfill(' ') << state.lennard_jones_energy_                             //
           << ", Field_sum_norm: " << std::setw(13) << std::setfill(' ') << sycl::length(state.forces_sum_)                //
           << ", Barycenter_speed_norm: " << std::setw(13) << std::setfill(' ') << state.compute_barycenter_speed_norm()   //
           << ", Avg_delta_energy: " << std::setprecision(5) << state.avg_delta_energy_                                    //
           << ", Time: " << state.configuration_.dt * state.simulation_idx_ << " fs"                                       //
           << ", Speed: " << std::setprecision(3) << 1.0 / state.avg_iter_duration_ << " iter/s.";
        return os;
    }
};


}   // namespace sim
