#!/bin/bash
# medicion de energia y potencia (apartado 4.5.4).
#
# este script NO se lanza con sbatch: se ejecuta con bash desde el front-end.
# como se vio en el laboratorio 1, con sbatch hay un problema de permisos que
# impide leer los contadores de energia; con salloc + srun si se puede medir.
#
# uso:
#   bash LanzaEnergia.sh matriz.bin :reposo + los 7 kernels
#   bash LanzaEnergia.sh matriz.bin kernel2opt1 kernel3:solo los indicados
#
# para cada kernel pide una reserva en stan (salloc) y dentro ejecuta todas sus
# combinaciones de hilos y repeticiones con srun + perf stat. se reserva una vez por
# kernel para no esperar en la cola en cada ejecucion y sin pasar de 10 minutos.

PARTICION=stan
NUCLEOS=16
HILOS="1 2 4 8 16"
REPETICIONES=3

# ---------------------------------------------------------------------------------
# modo interno: el script se llama a si mismo dentro de la reserva de salloc.
# va aqui arriba porque, en este modo, no debe ejecutarse nada de la parte de abajo
# ---------------------------------------------------------------------------------
if [ "$1" = "--medir" ]; then
    MATRIZ=$2
    k=$3
    FICHERO=$4

    # perf escribe aqui su resultado. va en output/ (carpeta compartida por nfs) y no
    # en /tmp, porque perf se ejecuta en el nodo y el script lee desde el front-end
    TMP="output/.perf_${k}_$$.txt"

    # mide una ejecucion y anade una linea al csv:
    #   ENERGIA,kernel,hilos,julios,segundos,vatios,repeticion
    # -a mide todo el procesador: la energia del paquete no se puede separar por
    #    proceso, asi que incluye tambien lo que consumen los nucleos que no usamos
    # -x, saca el resultado en formato csv, mas facil de procesar
    # duration_time da la duracion en ns para calcular la potencia = energia / tiempo
    medir() {
        local etiqueta=$1 h=$2 rep=$3
        shift 3
        export OMP_NUM_THREADS=$h
        srun -c "$NUCLEOS" perf stat -a -x, -e power/energy-pkg/,duration_time -o "$TMP" "$@" > /dev/null
        awk -F, -v k="$etiqueta" -v h="$h" -v rep="$rep" '
            $3 ~ /energy-pkg/  { e = $1 }
            $3 ~ /duration_time/ { t = $1 / 1e9 }
            END {
                # si perf no pudo leer el contador deja el texto tal cual ("<not supported>")
                if (e ~ /^[0-9.]+$/ && t > 0) printf "ENERGIA,%s,%s,%.3f,%.4f,%.3f,%s\n", k, h, e, t, e / t, rep
                else printf "ENERGIA,%s,%s,%s,%s,,%s\n", k, h, e, t, rep
            }' "$TMP" | tee -a "$FICHERO"
        rm -f "$TMP"
    }

    if [ "$k" = "reposo" ]; then
        # consumo del procesador sin ejecutar nada: sirve de referencia para saber
        # que parte de la energia medida se debe al kernel y cual al resto del chip
        for rep in $(seq 1 $REPETICIONES); do
            medir reposo 0 "$rep" sleep 5
        done
    elif [ "$k" = "kernel1" ]; then
        # kernel1 es secuencial: con mas hilos haria exactamente lo mismo
        for rep in $(seq 1 $REPETICIONES); do
            medir "$k" 1 "$rep" ./"$k" "$MATRIZ"
        done
    else
        for h in $HILOS; do
            for rep in $(seq 1 $REPETICIONES); do
                medir "$k" "$h" "$rep" ./"$k" "$MATRIZ"
            done
        done
    fi
    exit 0
fi

# ---------------------------------------------------------------------------------
# modo normal
# ---------------------------------------------------------------------------------
if [ "$#" -eq 0 ]; then
    echo "Error: No se ha proporcionado ningún argumento de entrada." >&2
    echo "Uso: bash $0 nombre_matriz [kernel1 kernel2 ...]" >&2
    exit 1
fi

MATRIZ=$1
shift
if [ "$#" -eq 0 ]; then
    KERNELS="reposo kernel1 kernel2 kernel3 kernel2opt1 kernel3opt1 kernel2opt2 kernel3opt2"
else
    KERNELS="$*"
fi

mkdir -p output
# un csv nuevo por lanzamiento, con fecha y hora para no machacar los anteriores
FICHERO="output/energia_$(date +%Y%m%d_%H%M%S).csv"
echo "tipo,kernel,hilos,julios,segundos,vatios,repeticion" > "$FICHERO"
echo "Matriz: $MATRIZ | Kernels: $KERNELS | Resultados en: $FICHERO"

# no modificar la compilacion de los archivos (lo exige el enunciado).
# se compila en el front-end: /home es compartido, asi que el nodo ve los binarios
for k in $KERNELS; do
    [ "$k" = "reposo" ] && continue
    gcc -o "$k" "$k.c" -lm -fopenmp -O0 || exit 1
done

for k in $KERNELS; do
    echo "--- $k: esperando reserva en $PARTICION ---"
    salloc -p "$PARTICION" -N 1 -c "$NUCLEOS" --time=00:10:00 \
        bash "$0" --medir "$MATRIZ" "$k" "$FICHERO"
done

echo "Terminado. Resultados en: $FICHERO"