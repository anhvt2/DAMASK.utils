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
# High-Level API C++ Makefile(.in)
# 

include $(top_srcdir)/config/commence.am

if BUILD_TESTS_CONDITIONAL
   TESTSERIAL_DIR =test
else
   TESTSERIAL_DIR=
endif

SUBDIRS=src $(TESTSERIAL_DIR)
DIST_SUBDIRS=src test examples

# Install examples
install-examples uninstall-examples:
	@(cd examples && $(MAKE) $(AM_MAKEFLAGS) $@) || exit 1;

installcheck-local:
	@(cd examples && $(MAKE) $(AM_MAKEFLAGS) $@) || exit 1;

# Clean examples when check-clean is invoked
check-clean ::
	@$(MAKE) $(AM_MAKEFLAGS) build-$@ || exit 1;
	@for d in X $(SUBDIRS) examples; do                                   \
	    if test $$d != X && test $$d != .; then                           \
	        (set -x; cd $$d && $(MAKE) $(AM_MAKEFLAGS) $@) || exit 1;     \
	    fi;                                                               \
	done

include $(top_srcdir)/config/conclude.am
