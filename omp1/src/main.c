#include <math.h>
#include <omp.h>
#include <stdio.h>

#define INITIAL_TRAPEZOIDS_NUMBER 1;

double *read_input(const char *filepath) {
    FILE *input_file = fopen(filepath, "r");

    if (input_file == NULL)
        return NULL;

    double a, b, error_val;
    fscanf(input_file, "%lf %lf %lf", &a, &b, &error_val);

    fclose(input_file);

    double arr[] = {a, b, error_val};
    return arr;
}

double f(double x) {
    return log(sin(x));
}

double calculate_integral_multi_thread(double a, double b, int num_trapezoids) {
    double sum = 0;
    double h = fabs(b - a) / num_trapezoids;

    #pragma omp parallel
    {
        double y = 0;

        #pragma omp for schedule(static)
        for (int i = 1; i < num_trapezoids; i++) {
            double x = a + i * h;
            y += f(x);
        }

        #pragma omp atomic
        sum += y;
    }

    return (h / 2) * (f(a) + f(b) + 2 * sum);
}

double calculate_integral_single_thread(double a, double b, int num_trapezoids) {
    double sum = 0;
    double h = fabs(b - a) / num_trapezoids;

    for (int i = 1; i < num_trapezoids; i++) {
        double x = a + i * h;
        sum += f(x);
    }

    return (h / 2) * (f(a) + f(b) + 2 * sum);
}

double calculate_integral(double a, double b, double error_val, int num_threads) {
    int num_trapezoids = 2 * INITIAL_TRAPEZOIDS_NUMBER;

    double prev_integral;
    double integral;

    error_val *= 3;

    if (num_threads == -1) {
        double start = omp_get_wtime();

        prev_integral = calculate_integral_single_thread(a, b, num_trapezoids / 2);
        integral = calculate_integral_single_thread(a, b, num_trapezoids);
        while (fabs(prev_integral - integral) > error_val) {
            num_trapezoids *= 2;
            prev_integral = integral;
            integral = calculate_integral_single_thread(a, b, num_trapezoids);
        }

        double end = omp_get_wtime();
        printf("Time consumed: %lf\n", end - start);

        return integral;
    }

    if (num_threads == 0)
        num_threads = omp_get_max_threads();

    omp_set_num_threads(num_threads);

    double start = omp_get_wtime();

    prev_integral = calculate_integral_multi_thread(a, b, num_trapezoids / 2);
    integral = calculate_integral_multi_thread(a, b, num_trapezoids);

    while ((fabs(prev_integral - integral)) > error_val) {
        num_trapezoids *= 2;
        prev_integral = integral;
        integral = calculate_integral_multi_thread(a, b, num_trapezoids);
    }

    double end = omp_get_wtime();
    printf("Time consumed: %lf\n", end - start);

    return integral;
}

void write_output(const char *filepath, double integral) {
    FILE *output_file = fopen(filepath, "w");
    fprintf(output_file, "%g\n", integral);
    fclose(output_file);
}

int main(int argc, const char * argv[]) {
    if (argc != 4) {
        printf("Command line arguments shall be <input_fp> <output_fp> <num_threads>\n");
        return 1;
    }

    const char *input_filepath = argv[1];
    const char *output_filepath = argv[2];
    int num_threads = atoi(argv[3]);

    if (num_threads < -1) {
        printf("Number of threads should be >= -1\n");
        return 1;
    }

    double *input = read_input(input_filepath);
    if (input == NULL) {
        printf("Input file not found!\n");
        return 1;
    }

    double a = input[0];
    double b = input[1];
    double error_val = input[2];

    double integral = calculate_integral(a, b, error_val, num_threads);

    write_output(output_filepath, integral);

    return 0;
}
