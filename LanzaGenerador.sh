#!/bin/bash
#SBATCH --job-name=ACProy
#SBATCH --partition=avignon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
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

mkdir -p output

echo Generando Matriz para el grupo $1
./GeneraMatriz $1
