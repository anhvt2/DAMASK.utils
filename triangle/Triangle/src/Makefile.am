## Process this file with automake to produce Makefile.in
AM_CFLAGS += @EXTRA_CFLAGS@

include_HEADERS = triangle.h

bin_PROGRAMS     = triangle showme
triangle_SOURCES =
##triangle_SOURCES = triangle.c triangle.h
##triangle_CFLAGS  = -DNO_PETSC_MALLOC
triangle_LDADD   = triangle.lo
showme_SOURCES   = showme.c

lib_LTLIBRARIES        = libtriangle.la
libtriangle_la_SOURCES =
##libtriangle_la_SOURCES = triangle.c triangle.h
##libtriangle_la_CFLAGS  = -DTRILIBRARY
libtriangle_la_LIBADD  = libtriangle.lo

triangle.lo: triangle.c triangle.h
	$(LTCOMPILE) -DNO_PETSC_MALLOC -c -o triangle.lo triangle.c

libtriangle.lo: triangle.c triangle.h
	$(LTCOMPILE) -DTRILIBRARY -c -o libtriangle.lo triangle.c
