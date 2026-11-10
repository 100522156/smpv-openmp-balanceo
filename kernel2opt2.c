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
// definiciones de perf_event_open (struct perf_event_attr, perf_type_hw_cache...),
// necesarias para leer los contadores de fallos de cache desde el propio programa
#include <linux/perf_event.h>


// =================================================================
// estructura de matriz csc y lectura
// =================================================================

typedef struct {
    int M;              // numero de filas
    int N;              // numero de columnas
    size_t nnz;         // numero total de elementos no nulos
    int *col_ptr;       // tamano n + 1
    int *row_ind;       // tamano nnz
    double *values;     // tamano nnz
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
// medicion de fallos de cache l1 con perf (por hilo)
// =================================================================
// el procesador tiene contadores internos que anotan eventos como "fallo en l1".
// se usan en tres pasos, como un cuentakilometros parcial:
//   abrir (elegir que contar) -> iniciar (poner a 0 y arrancar) -> parar (leer el valor).
// se hace desde el codigo y no con 'perf stat' desde fuera porque desde fuera
// solo se obtiene el total del programa, y aqui hace falta el dato de cada hilo.

// crea un contador de fallos de lectura en la cache l1 de datos para el hilo que
// la llama. 'static' porque es una funcion auxiliar que solo usa este archivo.
// devuelve el identificador del contador, o -1 si no se pudo crear.
static int perf_l1_abrir(void) {
    // formulario que describe a linux que se quiere contar. se pone entero a 0
    // primero porque un campo con basura puede hacer que se rechace o mida otra cosa
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));

    // tipo de evento: de cache (no ciclos ni instrucciones)
    attr.type = PERF_TYPE_HW_CACHE;
    // linux usa el tamano para saber que version de la estructura se le pasa
    attr.size = sizeof(attr);

    // que evento exacto: cache l1 de datos + operacion de lectura + resultado fallo.
    // se codifican los tres en un unico numero (bits 0-7, 8-15 y 16-23), que es
    // el formato que exige perf_event_open
    attr.config = PERF_COUNT_HW_CACHE_L1D |
                  (PERF_COUNT_HW_CACHE_OP_READ << 8) |
                  (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);

    // se crea parado: si contara desde ya, sumaria fallos de codigo que no es el kernel
    attr.disabled = 1;

    // solo se cuentan los fallos de nuestro programa, no los del sistema operativo
    // ni del hipervisor. ademas, sin ser administrador linux no deja medir esas partes
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;

    // pid = 0  -> mide al hilo que esta llamando a esta funcion. es lo que permite
    //             que cada hilo de openmp tenga su propio contador separado.
    // cpu = -1 -> en cualquier nucleo: si el so mueve el hilo, el contador le sigue.
    // group_fd = -1 y flags = 0 -> contador independiente, sin opciones extra.
    // no existe una funcion de c para esta llamada, por eso se usa syscall().
    return (int)syscall(SYS_perf_event_open, &attr, 0, -1, -1, 0);
}

// pone el contador a 0 y lo arranca justo antes de la zona que se quiere medir
static void perf_l1_iniciar(int fd) {
    // si no se pudo abrir, no se hace nada: el programa sigue y solo se pierde la medida
    if (fd < 0) return;
    // ioctl envia ordenes al contador a traves de su identificador
    ioctl(fd, PERF_EVENT_IOC_RESET, 0);   // poner a 0
    ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);  // empezar a contar
}

// para el contador, lee cuantos fallos anoto y lo cierra.
// devuelve long long porque los fallos pueden superar los ~2.100 millones que caben
// en un int, y con signo para poder usar -1 como "no se pudo medir" (un 0 se
// confundiria con "no hubo fallos")
static long long perf_l1_parar(int fd) {
    uint64_t fallos = 0;
    if (fd < 0) return -1;

    // se para antes de leer para que no siga sumando lo que viene despues del kernel
    ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);

    // el contador se lee como un fichero: devuelve un entero de 64 bits.
    // si la lectura falla se deja a 0 en vez de quedarse con un valor sin sentido
    if (read(fd, &fallos, sizeof(fallos)) != sizeof(fallos)) fallos = 0;

    // cada contador abierto ocupa un descriptor del sistema; hay que liberarlo
    close(fd);
    return (long long)fallos;
}


// =================================================================
// balanceo de carga con openmp (opt2)
// =================================================================
// problema del original: schedule(static) da a cada hilo un trozo seguido con el
// mismo numero de columnas, pero el coste real es el numero de no nulos de cada
// columna; si las columnas pesadas estan juntas, un hilo se lleva casi todo.
//
// en opt2 el enunciado solo permite cambiar o anadir llamadas de openmp, sin
// algoritmo propio. por eso respecto a kernel2.c solo cambia una cosa: la clausula
// schedule del bucle de columnas. no se calculan cortes ni se usa col_ptr para
// repartir; es openmp quien decide en tiempo de ejecucion que columnas hace cada hilo.
//
// se usa schedule(runtime) para que la politica de reparto se elija con la variable
// de entorno omp_schedule sin recompilar (por ejemplo "dynamic,64", "guided,64" o
// "static,64"). asi el script puede comparar varias politicas con el mismo binario;
// la que resulte mejor con la matriz del equipo se deja fija en la version final.

// 2. spmv openmp local con balanceo de openmp (opt2)
// igual que en kernel2.c, recibe dos parametros de salida para devolver lo medido
// por cada hilo (la medicion es la del apartado 4.4, no interviene en el reparto):
//   t_hilo[i]      -> segundos que ha tardado el hilo i
//   fallos_hilo[i] -> fallos de lectura en l1 del hilo i
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
        // numero de este hilo dentro del equipo (0, 1, 2...): indica en que posicion
        // de t_hilo y fallos_hilo debe dejar sus resultados
        int tid = omp_get_thread_num();

        // cada hilo abre su contador. como perf_l1_abrir usa pid=0, el contador
        // mide solo al hilo que lo abre y no se mezcla con los demas
        int fd = perf_l1_abrir();
        perf_l1_iniciar(fd);

        // cronometro propio del hilo. omp_get_wtime mide tiempo real en segundos,
        // que es lo que importa para ver si un hilo hace esperar a los demas
        double t_ini = omp_get_wtime();

        double *y_local = (double*)calloc(M, sizeof(double));

        // unico cambio respecto al original: schedule(runtime) en lugar de
        // schedule(static). con dynamic o guided cada hilo pide un nuevo paquete de
        // columnas al terminar el anterior, de modo que los hilos que reciben
        // columnas ligeras acaban haciendo mas paquetes y la carga se iguala sola.
        // el nowait se mantiene: cada hilo pasa al critical en cuanto no quedan
        // paquetes, sin esperar a los demas
        #pragma omp for schedule(runtime) nowait
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

        // se para aqui, despues del 'critical', y no justo tras el bucle: sumar
        // y_local en y tambien es trabajo del hilo, y si los hilos tienen que
        // esperarse para entrar en el 'critical', esa espera debe verse en su tiempo
        double t_fin = omp_get_wtime();
        long long fallos = perf_l1_parar(fd);

        // cada hilo escribe solo en su propia posicion [tid], asi que dos hilos
        // nunca escriben en el mismo sitio a la vez y no hay condicion de carrera
        t_hilo[tid] = t_fin - t_ini;
        fallos_hilo[tid] = fallos;

     }
}


// =================================================================
// evaluacion de benchmark
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



    // 2. ejecucion openmp local con balanceo de openmp

    // un hueco por hilo para que cada uno deje su tiempo y sus fallos.
    // calloc (y no malloc) para que empiecen a 0: si algun hilo no llegara a
    // ejecutarse, su linea saldria con 0 en vez de con un valor basura
    double *t_hilo = (double*)calloc(num_threads, sizeof(double));
    long long *fallos_hilo = (long long*)calloc(num_threads, sizeof(long long));

    // cronometro del kernel completo. rodea solo la llamada al kernel: la carga
    // del fichero ya se hizo antes y no debe contar (es lo que falseaba el tiempo
    // del programa entero en el apartado 4.3)
    double t_ini = omp_get_wtime();
    spmv_csc_openmp_local(mat, x, y_omp, t_hilo, fallos_hilo);
    double t_fin = omp_get_wtime();

    // se imprime fuera de la region paralela: si cada hilo hiciera su printf dentro,
    // las lineas saldrian desordenadas o mezcladas entre si.
    // formato csv con una etiqueta inicial (tiempo_kernel / hilo) para poder
    // filtrar estas lineas con grep desde el script y llevarlas a una tabla.
    //   tiempo_kernel,kernel,hilos,segundos
    //   hilo,kernel,hilos,id_hilo,segundos,fallos_l1
    printf("TIEMPO_KERNEL,kernel2opt2,%d,%.6f\n", num_threads, t_fin - t_ini);
    for (int i = 0; i < num_threads; i++) {
        printf("HILO,kernel2opt2,%d,%d,%.6f,%lld\n", num_threads, i, t_hilo[i], fallos_hilo[i]);
    }

    // ya impresos, los arrays de medidas no se necesitan
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
        printf(" Uso:  ./kernel2opt2  nombre_matriz \n\n");
    }

    return 0;
}