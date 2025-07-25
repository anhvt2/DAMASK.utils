#
# Copyright by The HDF Group.
# Copyright by the Board of Trustees of the University of Illinois.
# All rights reserved.
#
# This file is part of HDF5.  The full HDF5 copyright notice, including
# terms governing use, modification, and redistribution, is contained in
# the COPYING file, which can be found at the root of the source code
# distribution tree, or in https://www.hdfgroup.org/licenses.
# If you do not have access to either file, you may request a copy from
# help@hdfgroup.org.
##
## Makefile.am
## Run automake to generate a Makefile.in from this file.
#
# HDF5-C++ Makefile(.in)
#

include $(top_srcdir)/config/commence.am
include $(top_srcdir)/config/lt_vers.am

# Include src directory
AM_CPPFLAGS+=-I$(top_srcdir)/src

# This is our main target
lib_LTLIBRARIES=libhdf5_cpp.la

# Add libtool numbers to the HDF5 C++ library (from config/lt_vers.am)
libhdf5_cpp_la_LDFLAGS= -version-info $(LT_CXX_VERS_INTERFACE):$(LT_CXX_VERS_REVISION):$(LT_CXX_VERS_AGE) $(AM_LDFLAGS)

bin_SCRIPTS=h5c++

# Source files for the library
libhdf5_cpp_la_SOURCES=H5Exception.cpp H5IdComponent.cpp    \
    H5DataSpace.cpp H5PropList.cpp H5Library.cpp        	\
    H5FaccProp.cpp H5FcreatProp.cpp H5LaccProp.cpp      	\
	H5StrcreatProp.cpp H5LcreatProp.cpp H5AcreatProp.cpp 	\
    H5DxferProp.cpp H5DcreatProp.cpp H5Location.cpp     	\
    H5AbstractDs.cpp H5Attribute.cpp H5Object.cpp       	\
    H5OcreatProp.cpp H5DataType.cpp H5AtomType.cpp      	\
    H5PredType.cpp H5EnumType.cpp H5IntType.cpp     		\
    H5FloatType.cpp H5StrType.cpp H5ArrayType.cpp       	\
    H5VarLenType.cpp H5CompType.cpp H5DataSet.cpp       	\
    H5CommonFG.cpp H5Group.cpp H5File.cpp

# HDF5 C++ library depends on HDF5 Library.
libhdf5_cpp_la_LIBADD=$(LIBHDF5)

# Public headers
include_HEADERS=H5Cpp.h H5AbstractDs.h H5AtomType.h H5Attribute.h H5Classes.h \
    H5CommonFG.h H5CompType.h H5DataSet.h H5DataSpace.h H5DataType.h      \
    H5OcreatProp.h H5DcreatProp.h H5DxferProp.h H5EnumType.h              \
    H5Exception.h H5FaccProp.h H5FcreatProp.h H5File.h H5FloatType.h      \
    H5Group.h H5IdComponent.h H5Include.h H5IntType.h H5LaccProp.h        \
	H5StrcreatProp.h H5LcreatProp.h H5AcreatProp.h						  \
    H5Library.h H5Location.h H5Object.h H5PredType.h H5PropList.h         \
    H5StrType.h H5CppDoc.h H5ArrayType.h H5VarLenType.h

# h5c++ and libhdf5.settings are generated during configure.  Remove only when
# distclean.
DISTCLEANFILES=h5c++ libhdf5.settings

# Some C++ compilers/linkers (PGI?) create a directory named "ii_files" that
# holds *.ii files, which are template entity instantiations.
# This entire directory should be cleaned.
mostlyclean-local:
	@if test -d ii_files; then                                       \
	  $(RM) -rf ii_files;                                           \
	fi

# Mark this directory as part of the C++ API
CXX_API=yes

include $(top_srcdir)/config/conclude.am
