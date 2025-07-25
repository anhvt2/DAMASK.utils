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

if BUILD_PARALLEL_CONDITIONAL
  EXAMPLE_PROG_PARA = ph5example
endif

INSTALL_SCRIPT_FILES = run-c-ex.sh
INSTALL_TOP_SCRIPT_FILES = run-all-ex.sh
INSTALL_TOP_FILES = README

# Example programs.
# Don't tell automake about them, because if it knew they were programs,
# it would try to compile them instead of using the h5cc script.
# Use the boilerplate in config/examples.am instead.
EXAMPLE_PROG = h5_write h5_read h5_extend_write h5_chunk_read h5_compound \
           h5_crtgrpd h5_subset h5_cmprss h5_rdwt h5_crtgrpar h5_extend \
           h5_crtatt h5_crtgrp h5_crtdat \
           h5_group h5_select h5_attribute h5_mount h5_reference h5_drivers \
           h5_ref2reg h5_extlink h5_elink_unix2win h5_shared_mesg
TEST_SCRIPT=testh5cc.sh 
TEST_EXAMPLES_SCRIPT=$(INSTALL_SCRIPT_FILES)

# Install files
# List all file that should be installed in examples directory
INSTALL_FILES = h5_write.c h5_read.c h5_extend_write.c h5_chunk_read.c   \
          h5_crtgrpd.c h5_subset.c h5_cmprss.c h5_rdwt.c h5_crtgrpar.c \
          h5_extend.c h5_crtatt.c h5_crtgrp.c h5_crtdat.c \
          h5_compound.c h5_group.c h5_select.c h5_attribute.c h5_mount.c \
          h5_reference.c h5_drivers.c h5_extlink.c h5_elink_unix2win.c   \
          h5_ref2reg.c h5_shared_mesg.c ph5example.c



# How to build examples, using installed version of h5cc
if BUILD_PARALLEL_CONDITIONAL
$(EXTRA_PROG): $(H5CC_PP)
	$(H5CC_PP) $(H5CCFLAGS) $(CFLAGS) -o $@ $(srcdir)/$@.c;
else
$(EXTRA_PROG): $(H5CC)
	$(H5CC) $(H5CCFLAGS) $(CFLAGS) -o $@ $(srcdir)/$@.c;
endif

# Some examples depend on files created by other examples.
h5_read.chkexe_: h5_write.chkexe_
h5_chunk_read.chkexe_: h5_extend_write.chkexe_
h5_crtgrpd.chkexe_: h5_crtgrpar.chkexe_
# h5_rdwt and h5_crtatt both modify the same file created by
# h5_crtdat. Serialize them.
h5_rdwt.chkexe_: h5_crtdat.chkexe_
h5_crtatt.chkexe_: h5_rdwt.chkexe_

# The external link examples demonstrate how to use paths; they need
# directories to be created to do this.
EXTLINK_DIRS=red blue u2w

$(EXTLINK_DIRS):
	echo $(mkdir_p) $@
	$(mkdir_p) $@

CHECK_CLEANFILES+=$(EXTLINK_DIRS)

# Example directory
# Note: no '/' after DESTDIR.  Explanation in commence.am
EXAMPLEDIR=$(examplesdir)/c
EXAMPLETOPDIR=$(examplesdir)

# List dependencies for each program.  Normally, automake would take
# care of this for us, but if we tell automake about the programs it
# will try to build them with the normal C compiler, not h5cc.  This is
# an inelegant way of solving the problem.
# All programs share the same build rule and a dependency on the main hdf5
# library above.
h5_chunk_read: $(srcdir)/h5_chunk_read.c
h5_compound: $(srcdir)/h5_compound.c
h5_crtgrpd: $(srcdir)/h5_crtgrpd.c
h5_subset: $(srcdir)/h5_subset.c
h5_cmprss: $(srcdir)/h5_cmprss.c
h5_rdwt: $(srcdir)/h5_rdwt.c
h5_crtgrpar: $(srcdir)/h5_crtgrpar.c
h5_extend: $(srcdir)/h5_extend.c
h5_crtatt: $(srcdir)/h5_crtatt.c
h5_crtgrp: $(srcdir)/h5_crtgrp.c
h5_crtdat: $(srcdir)/h5_crtdat.c
h5_extend_write: $(srcdir)/h5_extend_write.c
h5_group: $(srcdir)/h5_group.c
h5_write: $(srcdir)/h5_write.c
h5_read: $(srcdir)/h5_read.c
h5_select: $(srcdir)/h5_select.c
h5_attribute: $(srcdir)/h5_attribute.c
h5_mount: $(srcdir)/h5_mount.c
h5_reference: $(srcdir)/h5_reference.c
h5_ref2reg: $(srcdir)/h5_ref2reg.c
h5_drivers: $(srcdir)/h5_drivers.c
ph5example: $(srcdir)/ph5example.c
h5_dtransform: $(srcdir)/h5_dtransform.c
h5_extlink: $(srcdir)/h5_extlink.c $(EXTLINK_DIRS)
h5_elink_unix2win: $(srcdir)/h5_elink_unix2win.c $(EXTLINK_DIRS)
h5_shared_mesg: $(srcdir)/h5_shared_mesg.c

if BUILD_SHARED_SZIP_CONDITIONAL
LD_LIBRARY_PATH=$(LL_PATH)
endif

include $(top_srcdir)/config/examples.am
include $(top_srcdir)/config/conclude.am
