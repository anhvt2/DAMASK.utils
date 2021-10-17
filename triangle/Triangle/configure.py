#!/usr/bin/env python
import user
import config.base
import os

class Configure(config.base.Configure):
  def __init__(self, framework):
    config.base.Configure.__init__(self, framework)
    self.headerPrefix = ''
    self.substPrefix  = ''
    self.foundMatlab  = 0
    self.compilers = self.framework.require('config.compilers', self)
    self.headers   = self.framework.require('config.headers', self)
    self.functions = self.framework.require('config.functions', self)
    self.libraries = self.framework.require('config.libraries', self)
    self.petscdir  = self.framework.require('PETSc.utilities.petscdir', self)
    self.X11       = self.framework.require('PETSc.packages.X11', self)
    self.petscdir.isPetsc = 0
    self.headers.headers.extend(['sys/time.h'])
    self.functions.functions.extend(['gettimeofday', 'strtod', 'strtol'])
    return

  def __str__(self):
    return ''
    
  def setupHelp(self, help):
    import nargs
    help.addArgument('Triangle', '-with-single=<bool>',           nargs.ArgBool(None, 0, 'Activate single precision arithmetic'))
    help.addArgument('Triangle', '-with-selfcheck=<bool>',        nargs.ArgBool(None, 0, 'Activate self checking code'))
    help.addArgument('Triangle', '-with-other-algorithms=<bool>', nargs.ArgBool(None, 1, 'Activate all meshing algorithms, not just constrained Delaunay'))
    help.addArgument('Triangle', '-with-research-code=<bool>',    nargs.ArgBool(None, 1, 'Turn this off to reduce binary size'))
    help.addArgument('Triangle', '-with-petsc=<bool>',            nargs.ArgBool(None, 0, 'Compile the Triangle library for use with PETSc'))
    return

  def configureSinglePrecision(self):
    '''By default, Triangle and Show Me use double precision floating point
    numbers.  If you prefer single precision, use the -DSINGLE switch.
    Double precision uses more memory, but improves the resolution of
    the meshes you can generate with Triangle.  It also reduces the
    likelihood of a floating exception due to overflow.  Also, it is
    much faster than single precision on 64-bit architectures like the
    DEC Alpha.  I recommend double precision unless you want to generate
    a mesh for which you do not have enough memory to use double precision.'''
    if self.framework.argDB['with-single']:
      self.addDefine('SINGLE', 1)
    return

  def configureSelfCheck(self):
    '''If you are modifying Triangle, I recommend turning on self checking while you are
    debugging.  This causes Triangle to include self-checking code.  Triangle will execute
    more slowly, however, so be sure to remove this switch before compiling a production version.'''
    if self.framework.argDB['with-selfcheck']:
      self.addDefine('SELF_CHECK', 1)
    return

  def configureAlgorithms(self):
    '''All meshing algorithms above and beyond constrained Delaunay triangulation
    may be disables.  Specifically, this eliminates the -r, -q, -a, -S, and -s switches.'''
    if not self.framework.argDB['with-other-algorithms']:
      self.addDefine('CDT_ONLY', 1)
    return

  def configureSmallBinary(self):
    '''If the size of the Triangle binary is important to you, you may wish to
    generate a reduced version of Triangle.  This gets rid of all features that
    are primarily of research interest.  Specifically, it eliminates the
    -i, -F, -s, and -C switches.'''
    if not self.framework.argDB['with-research-code']:
      self.addDefine('REDUCED', 1)
    return

  def configureTimer(self):
    if not self.headers.haveHeader('sys/time.h'):
      self.addDefine('NO_TIMER', 1)
    return

  def configure(self):
    self.addDefine('ANSI_DECLARATORS', 1)
    self.executeTest(self.configureSinglePrecision)
    self.executeTest(self.configureSelfCheck)
    self.executeTest(self.configureAlgorithms)
    self.executeTest(self.configureSmallBinary)
    self.executeTest(self.configureTimer)
    return

if __name__ == '__main__':
  import config.framework
  import sys

  framework = config.framework.Framework(sys.argv[1:])
  framework.addChild(Configure(framework))
  framework.configure()
  framework.dumpSubstitutions()
