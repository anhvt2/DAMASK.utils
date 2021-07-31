#!/bin/bash

function announce {
  echo -e "\n$(tput bold)$1$(tput sgr0)...\n"
}

PREFIX=/usr/local/opt
VERSION_HDF5=1.10.5
VERSION_PETSC=3.10.5
 
TMPDIR=/tmp
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
 
case $key in
--help|-h)
shift 1
echo "Usage: $(basename $0)"
for option in "--petsc $VERSION_PETSC" "--hdf5 $VERSION_HDF5" "--prefix $PREFIX"
do
  echo "       [$option]"
done
echo
echo "Purpose: ensure that prerequisites to run DAMASK under macOS are installed."
echo " * Xcode command line tools"
echo " * homebrew --> gcc + openmpi"
echo " * HDF5  [$VERSION_HDF5]"
echo " * PETSc [$VERSION_PETSC]"
exit
;;
--hdf5)
VERSION_HDF5="$2"
shift 2
;;
--petsc)
VERSION_PETSC="$2"
shift 2
;;
--prefix)
PREFIX="$2"
shift 2
;;
*)    # unknown option
POSITIONAL+=("$1") # save for later
shift
;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters
 
# ========================== #
#  Xcode Command Line Tools  #
# ========================== #

announce 'Xcode CLT'

[[ "$(xcode-select -p 1>/dev/null;echo $?)" == "0" ]] \
|| xcode-select --install

# ========================== #
#  Homebrew                  #
# ========================== #

announce 'Homebrew'

[[ "$(which brew)" == "" ]] \
&& /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

# ========================== #
#  Open MPI                  #
# ========================== #

announce 'Open MPI'

for pkg in openmpi
do
  me=${pkg%-source}
  [[ ${me}==${pkg} ]] && option='' || option='--build-from-source'
  [[ -d "$(brew --prefix $me 2>/dev/null)" ]] \
  || brew install $option $me
done

# ========================= #
#  HDF5                     #
# ========================= #

announce "HDF5 ${VERSION_HDF5}"

ME=hdf5
VERSION="${VERSION_HDF5}"
URL=http://www.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/${ME}-${VERSION}/src
 
wget -O $TMPDIR/${ME}-${VERSION}.tar.bz2 $URL/${ME}-${VERSION}.tar.bz2
 
[[ -d $TMPDIR/${ME}-${VERSION} ]] && rm -rf $TMPDIR/${ME}-${VERSION} 2>/dev/null
tar -C $TMPDIR -xjf $TMPDIR/${ME}-${VERSION}.tar.bz2
cd $TMPDIR/${ME}-${VERSION}
 
./configure \
--prefix=${PREFIX}/${ME}/${VERSION} \
--enable-parallel \
--enable-fortran \
--enable-build-mode=production \
FC=mpif90 \
CC=mpicc \
 
make
make install

# ========================== #
#  PETSc                     #
# ========================== #

announce "PETSc ${VERSION_PETSC}"

ME=petsc
VERSION="${VERSION_PETSC}"
URL=http://ftp.mcs.anl.gov/pub/petsc/release-snapshots

[[ -f $TMPDIR/${ME}-${VERSION}.tar.gz ]] \
|| wget -O $TMPDIR/${ME}-${VERSION}.tar.gz $URL/petsc-lite-${VERSION}.tar.gz
 
[[ -d $TMPDIR/${ME}-${VERSION} ]] && rm -rf $TMPDIR/${ME}-${VERSION} 2>/dev/null
tar -C $TMPDIR -xzf $TMPDIR/${ME}-${VERSION}.tar.gz
cd $TMPDIR/${ME}-${VERSION}
 
unset PETSC_DIR
unset PETSC_ARCH
 
./configure \
--prefix=${PREFIX}/${ME}/${VERSION} \
--with-hdf5-dir=${PREFIX}/hdf5/${VERSION_HDF5} \
--download-fftw \
--download-fblaslapack \
--download-scalapack \
--download-seacas \
--download-hypre \
--download-metis \
--download-parmetis \
--download-triangle \
--download-ml \
--download-mumps \
--download-suitesparse \
--download-superlu \
--download-superlu_dist \
--with-cxx-dialect=C++11 \
--with-c2html=0 \
--with-debugging=0 \
--with-ssl=0 \
--with-x=0 \
COPTFLAGS="-O3 -xHost" \
CXXOPTFLAGS="-O3 -xHost" \
FOPTFLAGS="-O3 -xHost" \
 
make PETSC_DIR=$(pwd) PETSC_ARCH=arch-darwin-c-opt all
make PETSC_DIR=$(pwd) PETSC_ARCH=arch-darwin-c-opt install
make PETSC_DIR=$(pwd) PETSC_ARCH=arch-darwin-c-opt test

