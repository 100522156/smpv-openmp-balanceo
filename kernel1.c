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
// definiciones de perf_event_open (struct perf_event_attr, PERF_TYPE_HW_CACHE...),
// necesarias para leer los contadores de fallos de cache desde el propio programa
#include <linux/perf_event.h>


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



// medicion de fallos de cache L1 con perf: tres funciones que son tres pasos
// (abrir, iniciar y parar), las mismas que en kernel2 y kernel3

// abrir crea un contador de fallos y devuelve su id, o -1 si no se pudo crear
static int perf_l1_abrir(void) {
    // creo la estructura del contador y la lleno de 0 para que no quede basura
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));
    // tipo de evento: cache
    attr.type = PERF_TYPE_HW_CACHE;
    // linux usa el tamaño para saber que version de la estructura le estamos pasando
    attr.size = sizeof(attr);
    // el evento: cache L1 de datos, lectura y fallo, codificado en un unico numero
    attr.config = PERF_COUNT_HW_CACHE_L1D |(PERF_COUNT_HW_CACHE_OP_READ << 8) |(PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
    // nace ya parado para que no cuente antes del kernel
    attr.disabled = 1;
    // solo nuestro programa, no el sistema operativo (kernel de linux) ni el hipervisor;
    // sin ser administrador linux tampoco deja medirlos
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;
    // pid=0: mide al hilo que llama; cpu=-1: en cualquier nucleo. se usa syscall()
    // porque no hay funcion de C para esta llamada
    return (int)syscall(SYS_perf_event_open, &attr, 0, -1, -1, 0);
}

// pone el contador a 0 y lo arranca. si no se pudo abrir no hace nada:
// se pierde la medida, no el calculo
static void perf_l1_iniciar(int fd) {
    if (fd < 0) return;
    ioctl(fd, PERF_EVENT_IOC_RESET, 0);   // poner a 0
    ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);  // empezar a contar
}

// para el contador, lee los fallos y lo cierra. long long porque pueden no caber en un int,
// y -1 si no se pudo medir, para no confundirlo con 0 fallos
static long long perf_l1_parar(int fd) {
    uint64_t fallos = 0;
    if (fd < 0) return -1;
    // se para antes de leer para no contar nada posterior al kernel
    ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
    // se lee como un fichero: un entero de 64 bits
    if (read(fd, &fallos, sizeof(fallos)) != sizeof(fallos)) fallos = 0;
    close(fd);
    return (long long)fallos;
}



// =================================================================
// KERNELS DE MULTIPLICACIÓN MATRIZ-VECTOR CON MEDIDA POR HILO
// =================================================================

// 1. SpMV Secuencial (1 hilo)
void spmv_csc_sequential(const CSCMatrix *mat, const double *x, double *y) {

    for (int i = 0; i < mat->M; i++) {
        y[i] = 0.0;
    }

    for (int j = 0; j < mat->N; j++) {
        double x_j = x[j];
        int start = mat->col_ptr[j];
        int end = mat->col_ptr[j + 1];

        for (int k = start; k < end; k++) {
            int row = mat->row_ind[k];
            for (int l = 1; l < 100; l++) {
                y[row] += mat->values[k] * x_j / l;
            }
        }
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

    // 1. Ejecución Secuencial

    // aqui no hay region paralela: un solo contador para el unico hilo.
    // si no se puede abrir se avisa, pero el programa sigue
    int fd = perf_l1_abrir();
    if (fd < 0) perror("Aviso: perf_event_open falló, no se medirán fallos L1");

    // cronometro del kernel: rodea solo la llamada, la carga del fichero no cuenta
    perf_l1_iniciar(fd);
    double t_ini = omp_get_wtime();

    spmv_csc_sequential(mat, x, y_seq);


    double t_fin = omp_get_wtime();
    long long fallos_l1 = perf_l1_parar(fd);

    // formato csv con etiqueta para filtrarlo con grep desde el script.
    // con un solo hilo, su tiempo es el del kernel
    printf("TIEMPO_KERNEL,kernel1,1,%.6f\n", t_fin - t_ini);
    printf("HILO,kernel1,1,0,%.6f,%lld\n", t_fin - t_ini, fallos_l1);


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
        printf(" Uso:  ./kernel1  nombre_matriz \n\n");
    }

    return 0;
}