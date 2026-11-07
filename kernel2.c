#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <omp.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <stdint.h>


// =================================================================
// ESTRUCTURA DE MATRIZ CSC Y LECTURA
// =================================================================

typedef struct {
    int M;              // Número de filas
    int N;              // Número de columnas
    size_t nnz;         // Número total de elementos no nulos
    int *col_ptr;       // Tamaño N + 1
    int *row_ind;       // Tamaño nnz
    double *values;     // Tamaño nnz
} CSCMatrix;

CSCMatrix* load_csc_matrix_bin(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        printf("Error: No se pudo abrir el archivo '%s'\n", filename);
        return NULL;
    }

    CSCMatrix *mat = (CSCMatrix*)malloc(sizeof(CSCMatrix));

    if (fread(&mat->M, sizeof(int), 1, f) != 1 ||
        fread(&mat->N, sizeof(int), 1, f) != 1 ||
        fread(&mat->nnz, sizeof(size_t), 1, f) != 1) {
        free(mat);
        fclose(f);
        return NULL;
    }

    mat->col_ptr = (int*)malloc((mat->N + 1) * sizeof(int));
    mat->row_ind = (int*)malloc(mat->nnz * sizeof(int));
    mat->values = (double*)malloc(mat->nnz * sizeof(double));

    fread(mat->col_ptr, sizeof(int), mat->N + 1, f);
    fread(mat->row_ind, sizeof(int), mat->nnz, f);
    fread(mat->values, sizeof(double), mat->nnz, f);

    fclose(f);
    return mat;
}

void free_csc_matrix(CSCMatrix *mat) {
    if (mat) {
        free(mat->col_ptr);
        free(mat->row_ind);
        free(mat->values);
        free(mat);
    }
}


// 2. SpMV OpenMP Local
void spmv_csc_openmp_local(const CSCMatrix *mat, const double *x, double *y) {
    int M = mat->M;
    int N = mat->N;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        y[i] = 0.0;
    }

    #pragma omp parallel
    {

        double *y_local = (double*)calloc(M, sizeof(double));

        #pragma omp for schedule(static) nowait
        for (int j = 0; j < N; j++) {
            double x_j = x[j];
            int start = mat->col_ptr[j];
            int end = mat->col_ptr[j + 1];

            for (int k = start; k < end; k++) {
                int row = mat->row_ind[k];
                for (int l = 1; l < 100; l++) {
                    y_local[row] += mat->values[k] * x_j / l;
                }
            }
        }

        #pragma omp critical
        {
            for (int i = 0; i < M; i++) {
                y[i] += y_local[i];
            }
        }

        free(y_local);

     }
}


// =================================================================
// EVALUACIÓN DE BENCHMARK
// =================================================================

void execute_benchmark(const char *filename) {
    CSCMatrix *mat = load_csc_matrix_bin(filename);
    if (!mat) return;

    int num_threads = omp_get_max_threads();

    printf("=========================================================\n");
    printf(" Procesando: %s\n", filename);
    printf(" Dimensiones: %d x %d | NNZ: %lu\n", mat->M, mat->N, mat->nnz);
    printf(" Hilos OpenMP disponibles: %d\n", num_threads);
    printf("=========================================================\n");

    double *x = (double*)malloc(mat->N * sizeof(double));
    double *y_seq = (double*)malloc(mat->M * sizeof(double));
    double *y_omp = (double*)malloc(mat->M * sizeof(double));

    for (int j = 0; j < mat->N; j++) {
        x[j] = 1.0;
    }



    // 2. Ejecución OpenMP Local
    spmv_csc_openmp_local(mat, x, y_omp);

    free(x);
    free(y_seq);
    free(y_omp);
    free_csc_matrix(mat);
}

int main(int argc, char *argv[]) {
    
    if (argc > 1) {
        execute_benchmark(argv[1]);
        return 0;
    }
    else{
        printf("\n Número de argumento incorrecto \n");
        printf(" Uso:  ./kernel2  nombre_matriz \n\n");
    }

    return 0;
}