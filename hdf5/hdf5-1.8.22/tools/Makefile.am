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
# Tools HDF5 Makefile(.in)
#

include $(top_srcdir)/config/commence.am

CONFIG=ordered

if BUILD_TESTS_CONDITIONAL
PERFORM_DIR=perform
else 
PERFORM_DIR=
endif 

# All subdirectories
SUBDIRS=lib h5diff h5ls h5dump misc h5import h5repack h5jam h5copy h5stat \
    $(PERFORM_DIR)

include $(top_srcdir)/config/conclude.am
