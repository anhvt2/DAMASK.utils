include make.inc

src/triangle.o: src/triangle.c src/triangle.h
	$(CC) $(CFLAGS) -o src/triangle.o -c src/triangle.c

$(TRIANGLELIB): src/triangle.c src/triangle.h
	$(CC) $(CFLAGS) -DNO_PETSC_MALLOC -DTRILIBRARY -o src/triangle.o -c src/triangle.c
	$(AR) $(ARFLAGS) $(TRIANGLELIB) src/triangle.o

clean:
	$(RM) triangle.o
