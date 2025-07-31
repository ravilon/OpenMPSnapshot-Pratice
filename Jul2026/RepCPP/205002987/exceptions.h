#pragma once
#include <exception>

namespace numath {

    /**
     * Exception for when the number of iterations runs out.
     */
    struct IterException : public std::exception {

        const char* what() const noexcept {
            return "Could not find anything with the specified number of iterations";
        }

    };

    /**
     * Exception for when the interval isn't valid.
     */
    struct IntervalException : public std::exception {

        const char* what() const noexcept {
            return "Invalid interval entered";
        }

    };

    /**
     * Exception for when the derivative is zero. 
     */
    struct DerivativeException : public std::exception {

        const char* what() const noexcept {
            return "Derivative equals 0. Possible multiple roots found";
        }

    };

    /**
     * Exception for when the denominator of the secant mehtod is zero. 
     */
    struct DenominatorException : public std::exception {

        const char* what() const noexcept {
            return "Denominator equals 0";
        }

    };

    /**
     * Exception for when there's more than one solution
     */
    struct SolutionException : public std::exception {

        const char* what() const noexcept {
            return "The system does not have an unique solution";
        }

    };

    /**
     * Exception for when a non existent method is called
     */
    struct MethodException : public std::exception {

        const char* what() const noexcept {
            return "The required methos does not exist";
        }

    };

}