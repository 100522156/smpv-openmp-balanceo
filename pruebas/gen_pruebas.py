# genera matrices de prueba en formato csc (.bin) con casos un poco extremos para comprobar que
# todas las versiones calculan el mismo resultado que kernel1.
# formato del .bin (el mismo que lee load_csc_matrix_bin):
#   int M, int N, size_t nnz, int col_ptr[N+1], int row_ind[nnz], double values[nnz]
#
# uso (desde la carpeta del proyecto): python3 pruebas/pruebas.py

import random
import struct
import os

CARPETA = os.path.dirname(os.path.abspath(__file__))


def escribir(nombre, M, N, columnas):
    # columnas: una lista por columna con las parejas (fila, valor) de sus no nulos
    col_ptr, row_ind, values = [0], [], []
    for col in columnas:
        # se ordenan las filas y se quitan las repetidas dentro de una misma columna
        vistas = {}
        for fila, valor in col:
            vistas[fila] = valor
        for fila in sorted(vistas):
            row_ind.append(fila)
            values.append(vistas[fila])
        # col_ptr guarda cuantos no nulos llevamos al terminar cada columna
        col_ptr.append(len(row_ind))

    nnz = len(row_ind)
    with open(os.path.join(CARPETA, nombre), "wb") as f:
        f.write(struct.pack("ii", M, N))
        f.write(struct.pack("Q", nnz))  # size_t en linux de 64 bits
        f.write(struct.pack("%di" % (N + 1), *col_ptr))
        if nnz:
            f.write(struct.pack("%di" % nnz, *row_ind))
            f.write(struct.pack("%dd" % nnz, *values))
    print("generada %-30s %7d x %-7d nnz=%d" % (nombre, M, N, nnz))


# semilla fija para que las matrices salgan siempre iguales
random.seed(1)

# 4x4 que se puede comprobar a mano
escribir("t1_pequena.bin", 4, 4,
         [[(0, 2.0), (2, 4.0)], [(1, 3.0)], [(0, 1.0), (2, 5.0)], [(3, 7.0)]])

# una sola fila y muchas columnas: todas las columnas van sumando en la misma posicion de y
escribir("t2_una_fila.bin", 1, 20000,
         [[(0, (-1.0 if j % 2 else 1.0) * (1.0 + j / 20000))]
          for j in range(20000)])

# una sola columna con muchas filas: casi todo el trabajo esta en una unica columna
escribir("t3_una_columna.bin", 50000, 1,
         [[(i, 1.0 / (i + 1)) for i in range(50000)]])

# matriz bastante grande pero vacia: el resultado tiene que salir todo a cero
escribir("t4_vacia_grande.bin", 100000, 80000,
         [[] for _ in range(80000)])

# muchas columnas vacias mezcladas con otras que tienen distinta cantidad de elementos
columnas = []
for j in range(20000):
    if j % 5 in (0, 1, 2):
        columnas.append([])
    else:
        cantidad = 1 + (j % 7)
        filas = random.sample(range(10000), cantidad)
        columnas.append([(fila, random.uniform(-2.0, 2.0)) for fila in filas])
escribir("t5_columnas_vacias.bin", 10000, 20000, columnas)

# todas las columnas suman en las mismas filas: asi se comprueba que no se pierdan sumas
columnas = []
for j in range(30000):
    columnas.append([
        (7, 1.0),
        (123, -0.25),
        (9999, 0.5),
        (random.randrange(10000), random.uniform(-0.1, 0.1))
    ])
escribir("t6_filas_compartidas.bin", 10000, 30000, columnas)

# matriz grande y desequilibrada: casi todo esta vacio y cuatro columnas tienen muchos elementos
columnas = [[] for _ in range(40000)]
for j in range(4):
    filas = random.sample(range(60000), 40000)
    columnas[j * 9000 + 100] = [(fila, random.uniform(-1.0, 1.0)) for fila in filas]
escribir("t7_grande_desequilibrada.bin", 60000, 40000, columnas)