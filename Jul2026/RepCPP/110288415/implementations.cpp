//
// Created by kirov on 13.11.17.
//
#include "implementations.h"
#include "omp.h"

using namespace std;

double f(double x) {
    return pow(sin(1 / x) / x, 2);
}

double J(double a, double b) {
    return 0.25 * (2 * (b - a) / (a * b) + sin(2 / b) - sin(2 / a));
}

bool break_condition(double a, double b, double eps) {
    return fabs(a - b) <= eps;
}

bool task_break_condition(double a, double b, double eps) {
    return fabs(a - b) <= eps * fabs(b);
}

Result serial_implementation(double a, double b, double eps) {
    double prev_result = (f(a) + f(b)) / 2;
    double step, result, start, finish;
    long step_number;
    long curr_step_number = 2;
    start = omp_get_wtime();

    while (true) {
        result = (f(a) + f(b)) / 2;
        step = (b - a) / curr_step_number;

        for (long i = 1; i < curr_step_number; i++) {
            double x = a + step * i;
            result += f(x);
        }
        result *= step;

        if (break_condition(prev_result, result, eps)) {
            break;
        } else {
            prev_result = result;
            curr_step_number *= 2;
        }
    }

    step_number = curr_step_number;

    finish = omp_get_wtime();

    return Result(1, finish - start, result, step_number);
}

Result serial_implementation_for_task(double a, double b, double eps) {
    double prev_result = (f(a) + f(b)) / 2;
    double step, result, start, finish;
    long step_number;
    long curr_step_number = 2;
    start = omp_get_wtime();

    while (true) {
        result = (f(a) + f(b)) / 2;
        step = (b - a) / curr_step_number;

        for (long i = 1; i < curr_step_number; i++) {
            double x = a + step * i;
            result += f(x);
        }
        result *= step;

        if (task_break_condition(prev_result, result, eps)) {
            break;
        } else {
            prev_result = result;
            curr_step_number *= 2;
        }
    }

    step_number = curr_step_number;

    finish = omp_get_wtime();

    return Result(1, finish - start, result, step_number);
}


Result reduction_implementation(double a, double b, double eps, int thread_number) {
    double prev_result = (f(a) + f(b)) / 2;
    double step, result, start, finish;
    long step_number;

    long long curr_step_number = 2;

    start = omp_get_wtime();

    while (true) {
        result = (f(a) + f(b)) / 2;
        step = (b - a) / curr_step_number;

#pragma omp parallel num_threads(thread_number)
#pragma omp for schedule(static) reduction(+:result)

        for (long i = 1; i < curr_step_number; i++) {
            double x = a + step * i;
            result += f(x);
        }
        result *= step;

        if (break_condition(prev_result, result, eps)) {
            break;
        } else {
            prev_result = result;
            curr_step_number *= 2;
        }
    }

    step_number = curr_step_number;

    finish = omp_get_wtime();

    return Result(thread_number, finish - start, result, step_number);
}

Result reduction_implementation_for_task(double a, double b, double eps, int thread_number) {
    double prev_result = (f(a) + f(b)) / 2;
    double step, result, start, finish;
    long step_number;

    long long curr_step_number = 2;

    start = omp_get_wtime();

    while (true) {
        result = (f(a) + f(b)) / 2;
        step = (b - a) / curr_step_number;

#pragma omp parallel num_threads(thread_number)
#pragma omp for schedule(static) reduction(+:result)

        for (long i = 1; i < curr_step_number; i++) {
            double x = a + step * i;
            result += f(x);
        }
        result *= step;

        if (task_break_condition(prev_result, result, eps)) {
            break;
        } else {
            prev_result = result;
            curr_step_number *= 2;
        }
    }

    step_number = curr_step_number;

    finish = omp_get_wtime();

    return Result(thread_number, finish - start, result, step_number);
}

Result atomic_implementation(double a, double b, double eps, int thread_number) {
    double prev_result = (f(a) + f(b)) / 2;
    double step, result, start, finish;
    long step_number;

    start = omp_get_wtime();

    long long curr_step_number = 2;

    while (true) {
        result = (f(a) + f(b)) / 2;
        step = (b - a) / curr_step_number;

#pragma omp parallel num_threads(thread_number)
#pragma omp for schedule(static)
        for (long i = 1; i < curr_step_number; i++) {
            double x = a + step * i;
#pragma omp atomic
            result += f(x);
        }

        result *= step;

        if (break_condition(prev_result, result, eps)) {
            break;
        } else {
            prev_result = result;
            curr_step_number *= 2;
        }
    }

    step_number = curr_step_number;

    finish = omp_get_wtime();

    return Result(thread_number, finish - start, result, step_number);
}

Result atomic_implementation_for_task(double a, double b, double eps, int thread_number) {
    double prev_result = (f(a) + f(b)) / 2;
    double step, result, start, finish;
    long step_number;

    start = omp_get_wtime();

    long long curr_step_number = 2;

    while (true) {
        result = (f(a) + f(b)) / 2;
        step = (b - a) / curr_step_number;

#pragma omp parallel num_threads(thread_number)
#pragma omp for schedule(static)
        for (long i = 1; i < curr_step_number; i++) {
            double x = a + step * i;
#pragma omp atomic
            result += f(x);
        }

        result *= step;

        if (task_break_condition(prev_result, result, eps)) {
            break;
        } else {
            prev_result = result;
            curr_step_number *= 2;
        }
    }

    step_number = curr_step_number;

    finish = omp_get_wtime();

    return Result(thread_number, finish - start, result, step_number);
}

Result critical_implementation(double a, double b, double eps, int thread_number) {
    long long curr_step_number = 2;
    double prev_result = (f(a) + f(b)) / 2;
    double step, result, start, finish;
    long step_number;

    start = omp_get_wtime();

    while (true) {
        result = (f(a) + f(b)) / 2;
        step = (b - a) / curr_step_number;

#pragma omp parallel num_threads(thread_number)
#pragma omp for schedule(static)
        for (long i = 1; i < curr_step_number; i++) {
            double x = a + step * i;
#pragma omp critical
            result += f(x);
        }

        result *= step;

        if (break_condition(prev_result, result, eps)) {
            break;
        } else {
            prev_result = result;
            curr_step_number *= 2;
        }
    }

    step_number = curr_step_number;

    finish = omp_get_wtime();

    return Result(thread_number, finish - start, result, step_number);
}

Result critical_implementation_for_task(double a, double b, double eps, int thread_number) {
    long long curr_step_number = 2;
    double prev_result = (f(a) + f(b)) / 2;
    double step, result, start, finish;
    long step_number;

    start = omp_get_wtime();

    while (true) {
        result = (f(a) + f(b)) / 2;
        step = (b - a) / curr_step_number;

#pragma omp parallel num_threads(thread_number)
#pragma omp for schedule(static)
        for (long i = 1; i < curr_step_number; i++) {
            double x = a + step * i;
#pragma omp critical
            result += f(x);
        }

        result *= step;

        if (task_break_condition(prev_result, result, eps)) {
            break;
        } else {
            prev_result = result;
            curr_step_number *= 2;
        }
    }

    step_number = curr_step_number;

    finish = omp_get_wtime();

    return Result(thread_number, finish - start, result, step_number);
}


Result locks_implementation(double a, double b, double eps, int thread_number) {
    double prev_result = (f(a) + f(b)) / 2;
    double step, result, start, finish;
    long step_number;
    long long curr_step_number = 2;

    start = omp_get_wtime();

    omp_lock_t writelock;

    omp_init_lock(&writelock);

    while (true) {
        result = (f(a) + f(b)) / 2;
        step = (b - a) / curr_step_number;

#pragma omp parallel num_threads(thread_number)
#pragma omp for schedule(static)
        for (long i = 1; i < curr_step_number; i++) {
            double x = a + step * i;

            omp_set_lock(&writelock);
            result += f(x);
            omp_unset_lock(&writelock);
        }

        result *= step;

        if (break_condition(prev_result, result, eps)) {
            break;
        } else {
            prev_result = result;
            curr_step_number *= 2;
        }
    }

    omp_destroy_lock(&writelock);

    step_number = curr_step_number;

    finish = omp_get_wtime();

    return Result(thread_number, finish - start, result, step_number);
}

Result locks_implementation_for_task(double a, double b, double eps, int thread_number) {
    double prev_result = (f(a) + f(b)) / 2;
    double step, result, start, finish;
    long step_number;
    long long curr_step_number = 2;

    start = omp_get_wtime();

    omp_lock_t writelock;

    omp_init_lock(&writelock);

    while (true) {
        result = (f(a) + f(b)) / 2;
        step = (b - a) / curr_step_number;

#pragma omp parallel num_threads(thread_number)
#pragma omp for schedule(static)
        for (long i = 1; i < curr_step_number; i++) {
            double x = a + step * i;

            omp_set_lock(&writelock);
            result += f(x);
            omp_unset_lock(&writelock);
        }

        result *= step;

        if (task_break_condition(prev_result, result, eps)) {
            break;
        } else {
            prev_result = result;
            curr_step_number *= 2;
        }
    }

    step_number = curr_step_number;

    finish = omp_get_wtime();

    return Result(thread_number, finish - start, result, step_number);
}