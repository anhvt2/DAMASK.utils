# $Id: makefile,v 1.11 2001/11/29 19:38:46 balay Exp balay $ 

########################################################################################
# Please read the readme file before editing the makefile
########################################################################################

ALL: blas_lib lapack_lib

########################################################################################
# Specify options to compile and create libraries
########################################################################################
FC         = f77
FOPTFLAGS  = -O
FNOOPT     = -O0
OMAKE      = make
RM         = /bin/rm
AR         = ar
AR_FLAGS   = cr
LIB_SUFFIX = a
RANLIB     = ranlib
########################################################################################
# By default, pick up the options from the PETSc configuration files
########################################################################################
BOPT             = O
BLASLAPACK_TYPE  = FBLASLAPACK
include ${PETSC_DIR}/bmake/common/base

########################################################################################
# compile the source files and create the blas and lapack libs
########################################################################################

BLAS_LIB_NAME       = libfblas.$(LIB_SUFFIX)
LAPACK_LIB_NAME     = libflapack.$(LIB_SUFFIX)
MAKE_OPTIONS        =  FC="$(FC)" FOPTFLAGS="$(FOPTFLAGS)" FNOOPT="$(FNOOPT)" AR="$(AR)" AR_FLAGS="$(AR_FLAGS)" RM="$(RM)"
MAKE_OPTIONS_BLAS   = $(MAKE_OPTIONS) LIBNAME="$(BLAS_LIB_NAME)"
MAKE_OPTIONS_LAPACK = $(MAKE_OPTIONS) LIBNAME="$(LAPACK_LIB_NAME)"

blas_lib:
	cd blas;   $(OMAKE) lib $(MAKE_OPTIONS_BLAS)
	$(RANLIB) $(BLAS_LIB_NAME)

lapack_lib:
	cd lapack; $(OMAKE) lib $(MAKE_OPTIONS_LAPACK)
	cd lapack-extra; $(OMAKE) lib $(MAKE_OPTIONS_LAPACK)
	$(RANLIB) $(LAPACK_LIB_NAME)

cleanblaslapck:
	$(RM) */*.o

cleanlib:
	$(RM) ./*.a ./*.lib

########################################################################################
# Target to create the fblaslapack distribution/tar file - using gnu-tar
########################################################################################
dist: cleanblaslapck cleanlib
	cd ..; 	/bin/rm -f fblaslapack.tar.gz; \
        /bin/tar --create --gzip --file fblaslapack.tar.gz --exclude=SCCS --exclude=BitKeeper --exclude=lib fblaslapack

#	/bin/tar cFFXf fblaslapack/maint/xclude - fblaslapack | gzip > fblaslapack.tar.gz
