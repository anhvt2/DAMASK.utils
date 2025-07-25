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
TEST_SCRIPT=testh5ls.sh
if HAVE_SHARED_CONDITIONAL
if USE_PLUGINS_CONDITIONAL
   TEST_SCRIPT += h5ls_plugin.sh
endif
endif
else 
TEST_SCRIPT=
endif
check_SCRIPTS=$(TEST_SCRIPT)
SCRIPT_DEPEND=h5ls$(EXEEXT)

# This is our main target, the h5ls tool
bin_PROGRAMS=h5ls

# Add h5ls specific linker flags here
h5ls_LDFLAGS = $(LT_STATIC_EXEC) $(AM_LDFLAGS)

# All programs depend on the hdf5 and h5tools libraries
LDADD=$(LIBH5TOOLS) $(LIBHDF5)

if NOT_UNAME_CYGWIN
if HAVE_SHARED_CONDITIONAL
  # Build it as shared library if configure is enabled for shared library.
  noinst_LTLIBRARIES=libdynlibls.la
  libdynlibls_la_SOURCES=dynlib_ls.c
  libdynlibls_la_LDFLAGS = -avoid-version -module -shared -export-dynamic -rpath /nowhere

libdynlibls.la: $(libdynlibls_la_OBJECTS) $(libdynlibls_la_DEPENDENCIES) $(EXTRA_libdynlibls_la_DEPENDENCIES)
	$(AM_V_CCLD)$(libdynlibls_la_LINK) $(am_libdynlibls_la_rpath) $(libdynlibls_la_OBJECTS) $(libdynlibls_la_LIBADD)
endif
endif

DISTCLEANFILES=h5ls_plugin.sh

include $(top_srcdir)/config/conclude.am
