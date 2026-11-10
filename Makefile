# makefile del proyecto de arquitectura de computadores (spmv csc + openmp).
# usa exactamente las mismas opciones de compilacion que los scripts originales
# (-lm -fopenmp -O0), ya que el enunciado no permite cambiarlas.
#
# uso:
#   make          -> compila los 7 kernels
#   make clean    -> borra los ejecutables

CC     = gcc
CFLAGS = -fopenmp -O0
LDLIBS = -lm

KERNELS = kernel1 kernel2 kernel3 kernel2opt1 kernel3opt1 kernel2opt2 kernel3opt2

all: $(KERNELS)

# regla comun: cada ejecutable se genera a partir de su .c con el mismo nombre
%: %.c
	$(CC) -o $@ $< $(LDLIBS) $(CFLAGS)

clean:
	rm -f $(KERNELS)

.PHONY: all clean