#!/bin/bash
#SBATCH --job-name=ACPruebas
#SBATCH --partition=stan
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00
#SBATCH --output=./output/pruebas_%j.out
#SBATCH --error=./output/pruebas_%j.err

# bateria de pruebas de correccion: comprueba que todas las versiones paralelas
# calculan el mismo vector y = A*x que kernel1 (secuencial), que se toma como referencia.
#
# uso (desde la carpeta del proyecto, despues de python3 pruebas/gen_pruebas.py):
#   sbatch pruebas/LanzaPruebas.sh                          -> las matrices pruebas/t*.bin
#   sbatch pruebas/LanzaPruebas.sh matriz_prueba1.bin ...   -> las matrices indicadas
# con la matriz del equipo conviene lanzarla sola para no pasar de 10 minutos.
#
# los kernels no se modifican: se crea una copia en pruebas/ con una linea anadida
# justo antes de free(x) que vuelca el vector y entero a un fichero de texto.

KERNELS="kernel2 kernel3 kernel2opt1 kernel3opt1 kernel2opt2 kernel3opt2"
# 3 hilos porque no divide el numero de columnas: prueba repartos desiguales
HILOS="1 2 3 4 8 16"
# error relativo maximo admitido. no se exige igualdad exacta porque al repartir
# la suma entre hilos cambia el orden de las sumas y con double eso cambia los
# ultimos decimales (la suma en coma flotante no es asociativa)
TOLERANCIA=1e-9

if [ "$#" -eq 0 ]; then
    MATRICES=$(ls pruebas/t*.bin 2>/dev/null)
    if [ -z "$MATRICES" ]; then
        echo "no hay matrices: ejecuta antes python3 pruebas/gen_pruebas.py" >&2
        exit 1
    fi
else
    MATRICES="$*"
fi

mkdir -p output
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# crea pruebas/t_<kernel>.c a partir de <kernel>.c y lo compila con las mismas
# opciones que el resto del proyecto. kernel1 guarda el resultado en y_seq y el
# resto en y_omp
preparar() {
    local k=$1 vec=y_omp
    [ "$k" = "kernel1" ] && vec=y_seq
    awk -v v="$vec" '
        /^    free\(x\);/ && !hecho {
            print "    { FILE *fp = fopen(getenv(\"PRUEBA_SALIDA\"), \"w\");"
            print "      for (int i = 0; i < mat->M; i++) fprintf(fp, \"%.17g\\n\", " v "[i]);"
            print "      fclose(fp); }"
            hecho = 1
        }
        { print }' "$k.c" > "pruebas/t_$k.c"
    gcc -o "pruebas/t_$k" "pruebas/t_$k.c" -lm -fopenmp -O0 || exit 1
}

# compara dos ficheros con un numero por linea; imprime el error relativo maximo
# (el error de cada elemento se divide entre max(1, |referencia|) para que los
# valores cercanos a 0 no den errores relativos enormes sin motivo)
comparar() {
    paste "$1" "$2" | awk '
        { d = $1 - $2; if (d < 0) d = -d
          r = $2; if (r < 0) r = -r; if (r < 1) r = 1
          if (d / r > m) m = d / r; n++ }
        END { if (n == 0) m = 0; printf "%.1e", m }'
}

preparar kernel1
for k in $KERNELS; do preparar "$k"; done

total=0
mal=0
for mat in $MATRICES; do
    echo "== $mat"
    # si la matriz no existe se avisa y se salta, para no confundirlo con un
    # kernel que calcula mal
    if [ ! -f "$mat" ]; then
        echo "   no existe, se omite"
        continue
    fi
    OMP_NUM_THREADS=1 PRUEBA_SALIDA="$TMP/ref" ./pruebas/t_kernel1 "$mat" > /dev/null 2>&1
    filas_ref=$(wc -l < "$TMP/ref")
    for k in $KERNELS; do
        linea="   $(printf '%-12s' "$k")"
        for h in $HILOS; do
            rm -f "$TMP/out"
            OMP_NUM_THREADS=$h PRUEBA_SALIDA="$TMP/out" ./pruebas/t_$k "$mat" > /dev/null 2>&1
            total=$((total + 1))
            if [ ! -f "$TMP/out" ] || [ "$(wc -l < "$TMP/out")" -ne "$filas_ref" ]; then
                linea="$linea  h=$h FALLO"
                mal=$((mal + 1))
                continue
            fi
            err=$(comparar "$TMP/out" "$TMP/ref")
            if awk -v e="$err" -v t="$TOLERANCIA" 'BEGIN { exit !(e <= t) }'; then
                linea="$linea  h=$h ok"
            else
                linea="$linea  h=$h MAL($err)"
                mal=$((mal + 1))
            fi
        done
        echo "$linea"
    done
done

echo "resultado: $((total - mal)) de $total pruebas correctas"
rm -f pruebas/t_kernel*