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
// Definiciones de perf_event_open (struct perf_event_attr, PERF_TYPE_HW_CACHE...),
// necesarias para leer los contadores de fallos de caché desde el propio programa
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


// =================================================================
// MEDICIÓN DE FALLOS DE CACHÉ L1 CON PERF (por hilo)
// =================================================================
// El procesador tiene contadores internos que anotan eventos como "fallo en L1".
// Se usan en tres pasos, como un cuentakilómetros parcial:
//   abrir (elegir qué contar) -> iniciar (poner a 0 y arrancar) -> parar (leer el valor).
// Se hace desde el código y no con 'perf stat' desde fuera porque desde fuera
// solo se obtiene el total del programa, y aquí hace falta el dato de cada hilo.

// Crea un contador de fallos de lectura en la caché L1 de datos para el hilo que
// la llama. 'static' porque es una función auxiliar que solo usa este archivo.
// Devuelve el identificador del contador, o -1 si no se pudo crear.
static int perf_l1_abrir(void) {
    // Formulario que describe a Linux qué se quiere contar. Se pone entero a 0
    // primero porque un campo con basura puede hacer que se rechace o mida otra cosa
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));

    // Tipo de evento: de caché (no ciclos ni instrucciones)
    attr.type = PERF_TYPE_HW_CACHE;
    // Linux usa el tamaño para saber qué versión de la estructura se le pasa
    attr.size = sizeof(attr);

    // Qué evento exacto: caché L1 de datos + operación de lectura + resultado fallo.
    // Se codifican los tres en un único número (bits 0-7, 8-15 y 16-23), que es
    // el formato que exige perf_event_open
    attr.config = PERF_COUNT_HW_CACHE_L1D |
                  (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                  (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);

    // Se crea parado: si contara desde ya, sumaría fallos de código que no es el kernel
    attr.disabled = 1;

    // Solo se cuentan los fallos de nuestro programa, no los del sistema operativo
    // ni del hipervisor. Además, sin ser administrador Linux no deja medir esas partes
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;

    // pid = 0  -> mide al hilo que está llamando a esta función. Es lo que permite
    //             que cada hilo de OpenMP tenga su propio contador separado.
    // cpu = -1 -> en cualquier núcleo: si el SO mueve el hilo, el contador le sigue.
    // group_fd = -1 y flags = 0 -> contador independiente, sin opciones extra.
    // No existe una función de C para esta llamada, por eso se usa syscall().
    return (int)syscall(SYS_perf_event_open, &attr, 0, -1, -1, 0);
}

// Pone el contador a 0 y lo arranca justo antes de la zona que se quiere medir
static void perf_l1_iniciar(int fd) {
    // Si no se pudo abrir, no se hace nada: el programa sigue y solo se pierde la medida
    if (fd < 0) return;
    // ioctl envía órdenes al contador a través de su identificador
    ioctl(fd, PERF_EVENT_IOC_RESET, 0);   // poner a 0
    ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);  // empezar a contar
}

// Para el contador, lee cuántos fallos anotó y lo cierra.
// Devuelve long long porque los fallos pueden superar los ~2.100 millones que caben
// en un int, y con signo para poder usar -1 como "no se pudo medir" (un 0 se
// confundiría con "no hubo fallos")
static long long perf_l1_parar(int fd) {
    uint64_t fallos = 0;
    if (fd < 0) return -1;

    // Se para antes de leer para que no siga sumando lo que viene después del kernel
    ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);

    // El contador se lee como un fichero: devuelve un entero de 64 bits.
    // Si la lectura falla se deja a 0 en vez de quedarse con un valor sin sentido
    if (read(fd, &fallos, sizeof(fallos)) != sizeof(fallos)) fallos = 0;

    // Cada contador abierto ocupa un descriptor del sistema; hay que liberarlo
    close(fd);
    return (long long)fallos;
}


// 2. SpMV OpenMP Local
// Respecto al original solo se añaden dos parámetros de salida. Las medidas se toman
// dentro de la función (cada hilo se mide a sí mismo), así que hace falta una forma
// de devolverlas a quien la llama:
//   t_hilo[i]      -> segundos que ha tardado el hilo i
//   fallos_hilo[i] -> fallos de lectura en L1 del hilo i
void spmv_csc_openmp_local(const CSCMatrix *mat, const double *x, double *y,
                           double *t_hilo, long long *fallos_hilo) {
    int M = mat->M;
    int N = mat->N;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        y[i] = 0.0;
    }

    #pragma omp parallel
    {
        // Número de este hilo dentro del equipo (0, 1, 2...): indica en qué posición
        // de t_hilo y fallos_hilo debe dejar sus resultados
        int tid = omp_get_thread_num();

        // Cada hilo abre SU contador. Como perf_l1_abrir usa pid=0, el contador
        // mide solo al hilo que lo abre y no se mezcla con los demás
        int fd = perf_l1_abrir();
        perf_l1_iniciar(fd);

        // Cronómetro propio del hilo. omp_get_wtime mide tiempo real en segundos,
        // que es lo que importa para ver si un hilo hace esperar a los demás
        double t_ini = omp_get_wtime();

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

        // Se para aquí, después del 'critical', y no justo tras el bucle: sumar
        // y_local en y también es trabajo del hilo, y si los hilos tienen que
        // esperarse para entrar en el 'critical', esa espera debe verse en su tiempo
        double t_fin = omp_get_wtime();
        long long fallos = perf_l1_parar(fd);

        // Cada hilo escribe solo en su propia posición [tid], así que dos hilos
        // nunca escriben en el mismo sitio a la vez y no hay condición de carrera
        t_hilo[tid] = t_fin - t_ini;
        fallos_hilo[tid] = fallos;

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

    // Un hueco por hilo para que cada uno deje su tiempo y sus fallos.
    // calloc (y no malloc) para que empiecen a 0: si algún hilo no llegara a
    // ejecutarse, su línea saldría con 0 en vez de con un valor basura
    double *t_hilo = (double*)calloc(num_threads, sizeof(double));
    long long *fallos_hilo = (long long*)calloc(num_threads, sizeof(long long));

    // Cronómetro del kernel completo. Rodea solo la llamada al kernel: la carga
    // del fichero ya se hizo antes y no debe contar (es lo que falseaba el tiempo
    // del programa entero en el apartado 4.3)
    double t_ini = omp_get_wtime();
    spmv_csc_openmp_local(mat, x, y_omp, t_hilo, fallos_hilo);
    double t_fin = omp_get_wtime();

    // Se imprime fuera de la región paralela: si cada hilo hiciera su printf dentro,
    // las líneas saldrían desordenadas o mezcladas entre sí.
    // Formato CSV con una etiqueta inicial (TIEMPO_KERNEL / HILO) para poder
    // filtrar estas líneas con grep desde el script y llevarlas a una tabla.
    //   TIEMPO_KERNEL,kernel,hilos,segundos
    //   HILO,kernel,hilos,id_hilo,segundos,fallos_L1
    printf("TIEMPO_KERNEL,kernel2,%d,%.6f\n", num_threads, t_fin - t_ini);
    for (int i = 0; i < num_threads; i++) {
        printf("HILO,kernel2,%d,%d,%.6f,%lld\n", num_threads, i, t_hilo[i], fallos_hilo[i]);
    }

    // Ya impresos, los arrays de medidas no se necesitan
    free(t_hilo);
    free(fallos_hilo);

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