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
#
#
## Makefile.am
## Run automake to generate a Makefile.in from this file.
#
# HDF5 High-Level Test Makefile(.in)
#

include $(top_srcdir)/config/commence.am

# Add include directories to C preprocessor flags
AM_CPPFLAGS+=-I. -I$(srcdir) -I$(top_builddir)/src -I$(top_srcdir)/src -I$(top_builddir)/test -I$(top_srcdir)/test -I$(top_srcdir)/hl/src

# The tests depend on the hdf5, hdf5 test,  and hdf5_hl libraries
LDADD=$(LIBH5_HL) $(LIBH5TEST) $(LIBHDF5)

# Test programs.  These are our main targets.  They should be listed in the
# order to be executed, generally most specific tests to least specific tests.
TEST_PROG=test_lite test_image test_file_image test_table test_ds test_packet test_dset_opt
check_PROGRAMS=$(TEST_PROG)

# These programs generate test files for the tests.  They don't need to be
# compiled every time we want to test the library.  However, putting
# them in a conditional causes automake to generate rules so that they
# can be built by hand.  They can also be built by specifying
# --enable-build-all at configure time.
BUILD_ALL_PROGS=gen_test_ds

if BUILD_ALL_CONDITIONAL
  noinst_PROGRAMS=$(BUILD_ALL_PROGS)
endif

# Temporary files.  These files are the ones created by running `make test'.
CHECK_CLEANFILES+=combine_tables[1-2].h5 test_ds[1-9].h5 test_ds10.h5 \
	test_image[1-3].h5 file_img[1-2].h5 test_lite[1-4].h5 test_table.h5 \
	test_packet_table.h5 test_packet_compress.h5 test_detach.h5 \
        test_dectris.h5 test_packet_table_vlen.h5 testfl_packet_table_vlen.h5

# Sources for test_packet executable
test_packet_SOURCES=test_packet.c test_packet_vlen.c

include $(top_srcdir)/config/conclude.am
