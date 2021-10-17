dnl PAC_CHECK_COMPILER_OPTION(optionname,action-if-ok,action-if-fail)
dnl This should actually check that compiler doesn't complain about it either,
dnl by compiling the same program with two options, and diff'ing the output.
dnl
define([PAC_CHECK_COMPILER_OPTION],[
AC_MSG_CHECKING([that C compiler accepts option $1])
CFLAGSSAV="$CFLAGS"
CFLAGS="$1 $CFLAGS"
echo 'void f(){}' > conftest.c
if test -z "`${CC-cc} $CFLAGS -c conftest.c 2>&1`"; then
  AC_MSG_RESULT(yes)
  $2
else
  AC_MSG_RESULT(no)
  $3
fi
rm -f conftest*
CFLAGS="$CFLAGSSAV"
])
