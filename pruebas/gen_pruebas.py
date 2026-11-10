# genera matrices pequenas en formato csc (.bin) con casos limite para comprobar que
# todas las versiones calculan el mismo resultado que kernel1.
# formato del .bin (el mismo que lee load_csc_matrix_bin):
#   int M, int N, size_t nnz, int col_ptr[N+1], int row_ind[nnz], double values[nnz]
#
# uso (desde la carpeta del proyecto): python3 pruebas/gen_pruebas.py

import random
import struct
import os

CARPETA = os.path.dirname(os.path.abspath(__file__))


def escribir(nombre, M, N, columnas):
    # columnas: una lista por columna con las parejas (fila, valor) de sus no nulos
    col_ptr, row_ind, values = [0], [], []
    for col in columnas:
        # se ordena por fila y se quitan filas repetidas en la misma columna,
        # porque en csc cada (fila, columna) aparece como mucho una vez
        vistas = {}
        for fila, valor in col:
            vistas[fila] = valor
        for fila in sorted(vistas):
            row_ind.append(fila)
            values.append(vistas[fila])
        # col_ptr guarda el acumulado de no nulos al final de cada columna
        col_ptr.append(len(row_ind))
    nnz = len(row_ind)
    with open(os.path.join(CARPETA, nombre), "wb") as f:
        f.write(struct.pack("ii", M, N))
        f.write(struct.pack("Q", nnz))  # size_t en linux de 64 bits
        f.write(struct.pack("%di" % (N + 1), *col_ptr))
        f.write(struct.pack("%di" % nnz, *row_ind))
        f.write(struct.pack("%dd" % nnz, *values))
    print("generada %-30s %6d x %-6d nnz=%d" % (nombre, M, N, nnz))


# semilla fija para que las matrices sean siempre las mismas
random.seed(1)

# 4x4 que se puede comprobar a mano
escribir("t1_pequena.bin", 4, 4,
         [[(0, 2.0), (2, 4.0)], [(1, 3.0)], [(0, 1.0), (2, 5.0)], [(3, 7.0)]])

# la mitad de las columnas vacias (col_ptr[j] == col_ptr[j+1])
escribir("t2_col_vacias.bin", 1000, 1000,
         [[(random.randrange(1000), random.random()) for _ in range(20)] if j % 2 == 0 else []
          for j in range(1000)])

# todo el trabajo en una sola columna: el peor caso para repartir por columnas
escribir("t3_una_columna.bin", 2000, 500,
         [[(i, 1.0) for i in range(2000)] if j == 250 else [] for j in range(500)])

# menos columnas que hilos: algunos hilos se quedan sin trabajo
escribir("t4_menos_cols_que_hilos.bin", 100, 3,
         [[(i, float(j + 1)) for i in range(0, 100, j + 1)] for j in range(3)])

# matriz sin ningun no nulo: y debe salir todo a 0
escribir("t5_vacia.bin", 10, 10, [[] for _ in range(10)])

# rectangular (muchas mas columnas que filas) y con valores negativos
escribir("t6_rectangular.bin", 300, 20000,
         [[(random.randrange(300), random.uniform(-1, 1)) for _ in range(random.randrange(0, 6))]
          for j in range(20000)])

# todas las columnas escriben en la fila 7: muchos hilos sumando en la misma y[7]
# a la vez, que es justo lo que protegen el critical y el atomic
escribir("t7_fila_compartida.bin", 50, 5000,
         [[(7, 1.0), (random.randrange(50), 0.5)] for j in range(5000)])