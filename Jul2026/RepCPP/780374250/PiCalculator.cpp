class PiCalculator {
private:
    long num_steps;

public:
    PiCalculator() : num_steps(500000) {}

    //   
    void setNumSteps(long steps) {
        num_steps = steps;
    }

    //     Pi
    double calculatePiSequential() {
        double step = 1.0 / static_cast<double>(num_steps);
        double pi = 0.0;

        for (int i = 0; i < num_steps; i++) {
            double x = (i + 0.5) * step;
            pi += 4.0 / (1.0 + x * x);
        }

        return step * pi;
    }

    //     Pi    for (static)
    double calculatePiParallelStatic() {
        double step = 1.0 / static_cast<double>(num_steps);
        double pi = 0.0;

        int i;
#pragma omp parallel for reduction(+:pi) schedule(static) private(i)
        for (i = 0; i < num_steps; i++) {
            double x = (i + 0.5) * step;
            pi += 4.0 / (1.0 + x * x);
        }

        return step * pi;
    }

    //     Pi    for (dynamic)
    double calculatePiParallelDynamic() {
        double step = 1.0 / static_cast<double>(num_steps);
        double pi = 0.0;

        int i;
#pragma omp parallel for reduction(+:pi) schedule(dynamic) private(i)
        for (i = 0; i < num_steps; i++) {
            double x = (i + 0.5) * step;
            pi += 4.0 / (1.0 + x * x);
        }

        return step * pi;
    }

    //     Pi    for (guided)
    double calculatePiParallelGuided() {
        double step = 1.0 / static_cast<double>(num_steps);
        double pi = 0.0;

        int i;
#pragma omp parallel for reduction(+:pi) schedule(guided) private(i)
        for (i = 0; i < num_steps; i++) {
            double x = (i + 0.5) * step;
            pi += 4.0 / (1.0 + x * x);
        }

        return step * pi;
    }
};