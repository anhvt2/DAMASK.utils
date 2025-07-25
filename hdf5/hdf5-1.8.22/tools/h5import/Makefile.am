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
# HDF5 Library Makefile(.in)
#

include $(top_srcdir)/config/commence.am

# Include src and tools/lib directories
AM_CPPFLAGS+=-I$(top_srcdir)/src -I$(top_srcdir)/tools/lib

# Test programs and scripts
if BUILD_TESTS_CONDITIONAL
	TEST_PROG=h5importtest
	TEST_SCRIPT=h5importtestutil.sh
else
	TEST_PROG=
	TEST_SCRIPT=
endif

check_PROGRAMS=$(TEST_PROG)
check_SCRIPT=h5importtestutil.sh
SCRIPT_DEPEND=h5import$(EXEEXT)

# Our main targets
bin_PROGRAMS=h5import

# Add h5import specific linker flags here
h5import_LDFLAGS = $(LT_STATIC_EXEC) $(AM_LDFLAGS)

# All programs depend on the main hdf5 library and the tools library
LDADD=$(LIBH5TOOLS) $(LIBHDF5)

# Temporary files from h5importtest
CHECK_CLEANFILES+=*.bin

include $(top_srcdir)/config/conclude.am
