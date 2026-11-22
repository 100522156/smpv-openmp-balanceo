#!/bin/bash
#SBATCH --job-name=ACProy
#SBATCH --partition=stan
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Sin esto Slurm reserva 1 núcleo y los hilos se reparten ese único núcleo
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00
#SBATCH --output=./output/kernels_%j.out
#SBATCH --error=./output/kernels_%j.err

# Uso:
#   sbatch LanzaKernels.sh matriz.bin:ejecuta kernel1, kernel2 y kernel3
#   sbatch LanzaKernels.sh matriz.bin kernel2 kernel3 : solo los kernels indicados
# Elegir kernels permite repartir las ejecuciones en varios trabajos si una sola
# tanda no cabe en el limite de 10 minutos por trabajo.

if [ "$#" -eq 0 ]; then
    echo "Error: No se ha proporcionado ningún argumento de entrada." >&2
    echo "Uso: $0 nombre_matriz [kernel1 kernel2 ...]" >&2
    exit 1
fi

MATRIZ=$1
shift # borro el argumento de la lista , pero ya he copiado el valor del primer argumento en matriz
# Si no se indica ningún kernel, se ejecutan los tres originales
if [ "$#" -eq 0 ]; then
    KERNELS="kernel1 kernel2 kernel3"
else
    KERNELS="$*"
fi

# varias repeticiones porque el nodo es compartido y una sola medida puede salir alterada

REPETICIONES=3
HILOS="1 2 4 8 16"

echo "Nodo: $(hostname) | Trabajo: $SLURM_JOB_ID | Matriz: $MATRIZ | Kernels: $KERNELS"

# No modificar la compilación de los archivos (lo exige el enunciado)
gcc -o kernel1 kernel1.c -lm -fopenmp -O0
gcc -o kernel2 kernel2.c -lm -fopenmp -O0
gcc -o kernel3 kernel3.c -lm -fopenmp -O0
# versiones optimizadas del apartado 4.5, con las mismas opciones de compilacion
gcc -o kernel2opt1 kernel2opt1.c -lm -fopenmp -O0
gcc -o kernel3opt1 kernel3opt1.c -lm -fopenmp -O0
gcc -o kernel2opt2 kernel2opt2.c -lm -fopenmp -O0
gcc -o kernel3opt2 kernel3opt2.c -lm -fopenmp -O0

# hago que time solo muestre el tiempo real en segundos del programa
TIMEFORMAT="%R"

# Fichero temporal donde se guarda la salida de cada ejecución para poder filtrarla.
# se borra con trap al terminar el script, acabe bien o mal
SALIDA=$(mktemp)
trap 'rm -f "$SALIDA" "$SALIDA.err"' EXIT

# Ejecuta un kernel una vez e imprime sus medidas.
# $1 = nombre del kernel, $2 = número de hilos, $3 = número de repetición
# Se imprimen tres tipos de línea, todas con la repetición al final:
#   PROGRAMA,kernel,hilos,segundos,rep :tiempo del programa entero (apartado 4.3)
#   TIEMPO_KERNEL,kernel,hilos,segundos,rep :tiempo solo del kernel (apartado 4.4)
#   HILO,kernel,hilos,id,segundos,fallos_L1,rep : datos de cada hilo (apartado 4.4)
ejecutar() {
    local k=$1 h=$2 rep=$3
    export OMP_NUM_THREADS=$h #para decirle a open cuantos hilos usare
    # La salida del programa va al fichero temporal y sus errores a otro, para que
    # no se mezclen con el numero que devuelve 'time'
    local t #como variable solo de la funcion
    #ejecuto el kernel con la matriz y guardo lo q me da el kernel(el tiempo del kernel y del hilo)
    t=$( { time ./"$k" "$MATRIZ" > "$SALIDA" 2> "$SALIDA.err"; } 2>&1 )
    echo "PROGRAMA,$k,$h,$t,$rep"
    # solo me interesan las lineas de medidas;uso tubarie para pasarselo a sed y cambiar el final de la linea $ por el numero de la repetcicion
    grep -E '^(TIEMPO_KERNEL|HILO),' "$SALIDA" | sed "s/\$/,$rep/"
    # Los avisos del programa (por ejemplo, si perf falla) se pasan al .err del trabajo
    cat "$SALIDA.err" >&2
}

for k in $KERNELS; do
    if [ "$k" = "kernel1" ]; then
        # kernel1 es secuencial, asi q no lo repito con mas hilos, solo lo hago 3 veces
        for rep in $(seq 1 $REPETICIONES); do
            ejecutar "$k" 1 "$rep"
        done
    else
        #cuando son los pareelos hago cada hilo 3 veces
        for h in $HILOS; do
            for rep in $(seq 1 $REPETICIONES); do
                ejecutar "$k" "$h" "$rep"
            done
        done
    fi
done