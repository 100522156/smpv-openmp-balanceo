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

if [ "$#" -eq 0 ]; then
    echo "Error: No se ha proporcionado ningún argumento de entrada." >&2
    echo "Uso: $0 nombre_matriz" >&2
    exit 1
fi

MATRIZ=$1
# Varias repeticiones porque stan es compartido y una sola medida puede salir alterada
REPETICIONES=3
HILOS="1 2 4 8 16"

echo "Nodo: $(hostname) | Trabajo: $SLURM_JOB_ID | Matriz: $MATRIZ"

# No modificar la compilación de los archivos (lo exige el enunciado)
gcc -o kernel1 kernel1.c -lm -fopenmp -O0
gcc -o kernel2 kernel2.c -lm -fopenmp -O0
gcc -o kernel3 kernel3.c -lm -fopenmp -O0

# Que bash imprima solo los segundos reales, así cada línea queda como CSV
TIMEFORMAT="%R"

echo "kernel,hilos,repeticion,tiempo_s"

# kernel1 es secuencial: repetirlo con más hilos no cambiaría nada
export OMP_NUM_THREADS=1
for rep in $(seq 1 $REPETICIONES); do
    t=$( { time ./kernel1 "$MATRIZ" > /dev/null; } 2>&1 )
    echo "kernel1,1,$rep,$t"
done

for k in kernel2 kernel3; do
    for h in $HILOS; do
        export OMP_NUM_THREADS=$h
        for rep in $(seq 1 $REPETICIONES); do
            t=$( { time ./$k "$MATRIZ" > /dev/null; } 2>&1 )
            echo "$k,$h,$rep,$t"
        done
    done
done