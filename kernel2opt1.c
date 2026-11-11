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
// (abrir, iniciar y parar). se hace desde el codigo y no con perf stat desde fuera
// porque desde fuera solo se obtiene el total del programa, y aqui hace falta el de cada hilo

// abrir crea un contador de fallos para el hilo que la llama y devuelve su id,
// o -1 si no se pudo crear
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

    // pid=0: mide al hilo que llama, asi cada hilo de openmp tiene su propio contador;
    // cpu=-1: en cualquier nucleo. se usa syscall() porque no hay funcion de C para esto
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


// balanceo de carga manual (opt1)
// problema del original: schedule(static) da a cada hilo las mismas columnas, pero el
// coste real es el numero de no nulos. si se concentran en una zona, el hilo que la
// recibe trabaja mucho mas y los demas le esperan.
// solucion: dividir las columnas en num_bloques trozos seguidos con mas o menos los
// mismos no nulos. como col_ptr[j] ya es el acumulado de no nulos de las columnas
// 0..j-1, basta con buscar en col_ptr donde se alcanza cada fraccion del total.
// solo usa col_ptr, asi que vale para cualquier matriz.
// devuelve corte[]: el bloque b son las columnas corte[b] .. corte[b+1]-1
// (corte[0] = 0 y corte[num_bloques] = N), o NULL si no hay memoria
static int* calcular_cortes(const CSCMatrix *mat, int num_bloques) {
    int N = mat->N;
    int *corte = (int*)malloc((num_bloques + 1) * sizeof(int));
    if (!corte) return NULL;

    corte[0] = 0;
    corte[num_bloques] = N;

    for (int b = 1; b < num_bloques; b++) {
        // no nulos que deberian quedar antes del bloque b en un reparto perfecto.
        // en size_t porque nnz * b puede no caber en un int
        size_t objetivo = (mat->nnz * (size_t)b) / (size_t)num_bloques;

        // busqueda binaria de la primera columna j con col_ptr[j] >= objetivo.
        // se puede porque col_ptr nunca decrece; cuesta O(log N), despreciable
        int izq = 0, der = N;
        while (izq < der) {
            int mitad = izq + (der - izq) / 2;
            if ((size_t)mat->col_ptr[mitad] < objetivo) izq = mitad + 1;
            else der = mitad;
        }
        int j = izq;

        // el objetivo suele caer en mitad de una columna: se corta en el borde
        // (antes o despues) que queda mas cerca, para no cargar siempre al bloque anterior
        if (j > 0 && objetivo - (size_t)mat->col_ptr[j - 1] < (size_t)mat->col_ptr[j] - objetivo) {
            j = j - 1;
        }

        // los cortes no pueden retroceder: con columnas enormes dos objetivos pueden
        // caer en la misma columna y entonces ese bloque queda vacio (es correcto)
        if (j < corte[b - 1]) j = corte[b - 1];
        corte[b] = j;
    }
    return corte;
}


// 2. SpMV OpenMP Local con balanceo manual (opt1)
// cambios respecto a kernel2.c: el parametro num_bloques (se usa el numero de hilos,
// que execute_benchmark ya obtenia en el original) y que el bucle repartido por openmp
// recorre bloques en vez de columnas. los #pragma son los del original, como exige
// el enunciado para opt1. t_hilo y fallos_hilo devuelven lo que mide cada hilo
void spmv_csc_openmp_local(const CSCMatrix *mat, const double *x, double *y,int num_bloques, double *t_hilo, long long *fallos_hilo) {
    int M = mat->M;

    // los cortes se calculan dentro del kernel para que su coste cuente en el tiempo
    // medido y la comparacion con el original sea justa
    int *corte = calcular_cortes(mat, num_bloques);
    if (!corte) {
        printf("Error: sin memoria para los cortes de balanceo\n");
        return;
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        y[i] = 0.0;
    }

    #pragma omp parallel
    {
        // numero de este hilo (0, 1, 2...): su posicion en t_hilo y fallos_hilo
        int tid = omp_get_thread_num();

        // cada hilo abre su contador; con pid=0 mide solo al hilo que lo abre
        int fd = perf_l1_abrir();
        perf_l1_iniciar(fd);

        // cronometro propio del hilo, en tiempo real: lo que importa para ver
        // si un hilo hace esperar a los demas
        double t_ini = omp_get_wtime();

        // copia privada de y, ya a 0 (calloc): al escribir cada hilo en la suya no hay
        // condicion de carrera. va despues del cronometro: reservarla tambien es coste del hilo
        double *y_local = (double*)calloc(M, sizeof(double));

        // mismo pragma que el original, pero ahora reparte bloques: con tantos bloques
        // como hilos, schedule(static) da exactamente uno a cada hilo.
        // nowait: el hilo que acaba pasa al critical sin esperar a los demas
        #pragma omp for schedule(static) nowait
        for (int b = 0; b < num_bloques; b++) {
            // recorre las columnas de su bloque; el calculo es el original
            for (int j = corte[b]; j < corte[b + 1]; j++) {
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
        }

        #pragma omp critical
        {
            for (int i = 0; i < M; i++) {
                y[i] += y_local[i];
            }
        }

        free(y_local);

        // se para despues del critical y no justo tras el bucle: volcar y_local en y
        // tambien es trabajo del hilo, y la espera para entrar debe verse en su tiempo
        double t_fin = omp_get_wtime();
        long long fallos = perf_l1_parar(fd);

        // cada hilo escribe solo en su posicion [tid]: no hay condicion de carrera
        t_hilo[tid] = t_fin - t_ini;
        fallos_hilo[tid] = fallos;

     }

    free(corte);
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



    // 2. Ejecución OpenMP Local con balanceo manual

    // un hueco por hilo para su tiempo y sus fallos. calloc para que empiecen a 0
    // y no salga basura si algun hilo no llegara a escribir
    double *t_hilo = (double*)calloc(num_threads, sizeof(double));
    long long *fallos_hilo = (long long*)calloc(num_threads, sizeof(long long));

    // cronometro del kernel: rodea solo la llamada, la carga del fichero no cuenta.
    // num_threads se pasa como numero de bloques: un bloque equilibrado por hilo
    double t_ini = omp_get_wtime();
    spmv_csc_openmp_local(mat, x, y_omp, num_threads, t_hilo, fallos_hilo);
    double t_fin = omp_get_wtime();

    // se imprime fuera de la region paralela para que las lineas no salgan mezcladas.
    // formato csv con etiqueta para filtrarlo con grep desde el script:
    //   TIEMPO_KERNEL,kernel,hilos,segundos
    //   HILO,kernel,hilos,id_hilo,segundos,fallos_L1
    printf("TIEMPO_KERNEL,kernel2opt1,%d,%.6f\n", num_threads, t_fin - t_ini);
    for (int i = 0; i < num_threads; i++) {
        printf("HILO,kernel2opt1,%d,%d,%.6f,%lld\n", num_threads, i, t_hilo[i], fallos_hilo[i]);
    }

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
        printf(" Uso:  ./kernel2opt1  nombre_matriz \n\n");
    }

    return 0;
}