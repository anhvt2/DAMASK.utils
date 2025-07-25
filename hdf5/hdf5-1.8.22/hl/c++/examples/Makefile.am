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
##
#
# HDF5 Library Examples Makefile(.in)
#

include $(top_srcdir)/config/commence.am

# These are the programs that 'make all' or 'make prog' will build and
# which 'make check' will run.  List them in the order they should be run.
EXAMPLE_PROG=ptExampleFL
TEST_EXAMPLES_SCRIPT=$(INSTALL_SCRIPT_FILES)

# These are the example files to be installed
INSTALL_FILES=ptExampleFL.cpp
INSTALL_SCRIPT_FILES = run-hlc++-ex.sh

# Tell conclude.am that these are C++ tests.
CXX_API=yes

# Where to install examples
# Note: no '/' after DESTDIR.  Explanation in commence.am
EXAMPLEDIR=$(examplesdir)/hl/c++
EXAMPLETOPDIR=$(examplesdir)/hl

# How to build programs using h5c++
$(EXTRA_PROG): $(H5CPP)
	$(H5CPP) $(H5CCFLAGS) $(CPPFLAGS) -o $@ $(srcdir)/$@.cpp

# List dependencies for each program.  Normally, automake would take
# care of this for us, but if we tell automake about the programs it
# will try to build them with the normal C++ compiler, not h5c++.  This is
# an inelegant way of solving the problem, unfortunately.
ptExampleFL: ptExampleFL.cpp

include $(top_srcdir)/config/examples.am
include $(top_srcdir)/config/conclude.am
