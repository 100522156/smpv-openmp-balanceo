#!/bin/bash
#SBATCH --job-name=ACProy
#SBATCH --partition=stan
#SBATCH --nodes=1
#SBATCH --ntasks=1
# sin esto slurm reserva 1 nucleo y los hilos se reparten ese unico nucleo
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00
#SBATCH --output=./output/schedules_%j.out
#SBATCH --error=./output/schedules_%j.err

# pruebas para elegir el schedule de las versiones opt2 (apartado 4.5.2).
# los kernels opt2 usan schedule(runtime), asi que el reparto lo decide la variable
# OMP_SCHEDULE; este script ejecuta el mismo binario con varias politicas y anota
# los resultados para quedarse con la mas rapida.
#
# uso:
#   sbatch LanzaSchedules.sh matriz.bin :kernel2opt2 y kernel3opt2
#   sbatch LanzaSchedules.sh matriz.bin kernel3opt2 :solo los indicados
# elegir kernels permite repartir las pruebas en varios trabajos si no caben en 10 min.

if [ "$#" -eq 0 ]; then
    echo "Error: No se ha proporcionado ningún argumento de entrada." >&2
    echo "Uso: $0 nombre_matriz [kernel2opt2 kernel3opt2]" >&2
    exit 1
fi

MATRIZ=$1
shift
if [ "$#" -eq 0 ]; then
    KERNELS="kernel2opt2 kernel3opt2"
else
    KERNELS="$*"
fi

# politicas a comparar:
#   static: el reparto del original, sirve de referencia
#   static,64 : paquetes de 64 columnas por turnos, sin coordinacion
#   dynamic,N : >el hilo que acaba coge el siguiente paquete; varios n para ver
#                el efecto del tamano (pequeno = mas coordinacion, grande = peor reparto)
#   guided,N : como dynamic pero con paquetes que empiezan grandes y van bajando
SCHEDULES="static static,64 dynamic,16 dynamic,64 dynamic,256 guided,16 guided,64"
# varios numeros de hilos para ver si la politica ganadora cambia con los hilos
HILOS="4 8 16"
# varias repeticiones porque el nodo es compartido y una sola medida puede salir alterada
REPETICIONES=3

echo "Nodo: $(hostname) | Trabajo: $SLURM_JOB_ID | Matriz: $MATRIZ | Kernels: $KERNELS"

# no modificar la compilacion de los archivos (lo exige el enunciado)
for k in $KERNELS; do
    gcc -o "$k" "$k.c" -lm -fopenmp -O0
done

# que bash imprima solo los segundos reales del programa completo
TIMEFORMAT="%R"

# fichero temporal para la salida de cada ejecucion; se borra al terminar
SALIDA=$(mktemp)
trap 'rm -f "$SALIDA" "$SALIDA.err"' EXIT

# ejecuta un kernel una vez con un schedule concreto e imprime sus medidas.
# a cada linea se le anade al final la repeticion y el schedule. en el schedule la
# coma se cambia por ':' (dynamic:64) para que no rompa las columnas del csv
ejecutar() {
    local k=$1 h=$2 rep=$3 sched=$4
    export OMP_NUM_THREADS=$h
    export OMP_SCHEDULE=$sched
    local etiqueta=${sched/,/:}
    local t
    t=$( { time ./"$k" "$MATRIZ" > "$SALIDA" 2> "$SALIDA.err"; } 2>&1 )
    echo "PROGRAMA,$k,$h,$t,$rep,$etiqueta"
    grep -E '^(TIEMPO_KERNEL|HILO),' "$SALIDA" | sed "s/\$/,$rep,$etiqueta/"
    cat "$SALIDA.err" >&2
}

for k in $KERNELS; do
    for h in $HILOS; do
        for sched in $SCHEDULES; do
            for rep in $(seq 1 $REPETICIONES); do
                ejecutar "$k" "$h" "$rep" "$sched"
            done
        done
    done
done