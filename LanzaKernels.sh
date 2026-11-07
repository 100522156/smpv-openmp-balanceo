#!/bin/bash
#SBATCH --job-name=ACProy
#SBATCH --partition=avignon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --output=./output/info_nodo_%j.out
#SBATCH --error=./output/info_nodo_%j.err

echo "=================================================="
echo "INFORMACIÓN DEL NODO ASIGNADO"
echo "=================================================="
echo "Nodo de ejecución : $(hostname)"
echo "ID del Trabajo     : $SLURM_JOB_ID"
echo "=================================================="
echo ""

# Verificar si el número de argumentos ($#) es igual a 0
if [ "$#" -eq 0 ]; then
    echo "Error: No se ha proporcionado ningún argumento de entrada." >&2
    echo "Uso: $0 nombre_matriz" >&2
    exit 1
fi

export OMP_NUM_THREADS=4

# No modificar la compilación de los archivos
gcc -o kernel1 kernel1.c -lm -fopenmp -O0
gcc -o kernel2 kernel2.c -lm -fopenmp -O0
gcc -o kernel3 kernel3.c -lm -fopenmp -O0


# Ejecución de los kernels
 ./kernel1 $1
 ./kernel2 $1
 ./kernel3 $1

