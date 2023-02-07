#include <omp.h>
#include <stdbool.h>
#include <stdio.h>

#define HISTOGRAM_LENGTH 256

typedef struct Picture {
    uint32_t max_brightness;
    uint32_t length;
    unsigned char * brightnesses;
} Picture;

void destruct_picture(Picture picture) {
    free(picture.brightnesses);
}

Picture read_input(const char* filepath) {
    FILE* input_file = fopen(filepath, "rb");

    Picture picture;
    if (input_file == NULL) {
        printf("Input file not found!\n");
        exit(1);
    }

    uint32_t width, height;
    fscanf(input_file, "P5\n%d %d\n%d", &width, &height, &picture.max_brightness);
    fgetc(input_file);

    picture.length = height * width;

    picture.brightnesses = malloc(picture.length * sizeof(unsigned char *));
    fread(picture.brightnesses, sizeof(unsigned char), picture.length, input_file);

    fclose(input_file);

    return picture;
}

void calculate_histogram_single_thread(Picture picture, uint32_t histogram[]) {
    for (int i = 0; i < picture.length; i++) {
        histogram[picture.brightnesses[i]]++;
    }
}

void calculate_histogram_multi_thread(Picture picture, uint32_t histogram[]) {
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < picture.length; i++) {
            #pragma omp atomic
            histogram[picture.brightnesses[i]]++;
        }
    }
}

void calculate_histogram(Picture picture, int num_threads, uint32_t histogram[]) {
    if (num_threads == -1) {
        double start = omp_get_wtime();
        calculate_histogram_single_thread(picture, histogram);
        printf("Time (%i thread(s)): %g ms\n", 1, omp_get_wtime() - start);
        return;
    }
    if (num_threads == 0) {
        num_threads = omp_get_max_threads();
    }

    omp_set_num_threads(num_threads);

    double start = omp_get_wtime();
    calculate_histogram_multi_thread(picture, histogram);
    printf("Time (%i thread(s)): %g ms\n", num_threads, omp_get_wtime() - start);
}

void write_output(const char* filepath, uint32_t histogram[]) {
    FILE* output_file = fopen(filepath, "wb");
    if (output_file == NULL) {
        printf("Output file not found!\n");
        exit(1);
    }
    fwrite(histogram, sizeof(uint32_t), HISTOGRAM_LENGTH, output_file);
    fclose(output_file);
}

void release_resources(Picture picture) {
    destruct_picture(picture);
}

int main(int argc, const char* argv[]) {
    if (argc != 4) {
        printf("Command line arguments shall be <input_fp> <output_fp> <num_threads>\n");
        return 1;
    }

    const char* input_filepath = argv[1];
    const char* output_filepath = argv[2];
    int num_threads = atoi(argv[3]);

    if (num_threads < -1) {
        printf("Number of threads should be >= -1\n");
        return 1;
    }

    Picture picture = read_input(input_filepath);

    uint32_t histogram[HISTOGRAM_LENGTH] = {0};
    calculate_histogram(picture, num_threads, histogram);
    release_resources(picture);

    write_output(output_filepath, histogram);

    return 0;
}
