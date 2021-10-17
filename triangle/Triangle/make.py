#!/usr/bin/env python
import user
import maker

import os

class Make(maker.Make):
  def setupDependencies(self, sourceDB):
    maker.Make.setupDependencies(self, sourceDB)
    sourceDB.addDependency(os.path.join('src', 'triangle.c'), os.path.join('src', 'triangle.h'))
    return

  def setupConfigure(self, framework):
    ret = maker.Make.setupConfigure(self, framework)
    framework.header = os.path.join('src', 'config.h')
    framework.cHeader = ''
    return ret

  def configure(self, builder):
    framework = maker.Make.configure(self, builder)
    self.setCompilers = framework.require('config.setCompilers', None)
    self.libraries = framework.require('config.libraries', None)
    self.X11= framework.require('PETSc.packages.X11', None)
    return

  def buildDirectories(self, builder):
    if self.prefix is None:
      self.includeDir = 'include'
      self.libDir     = 'lib'
      self.binDir     = 'bin'
    else:
      self.includeDir = os.path.join(self.prefix, 'include')
      self.libDir     = os.path.join(self.prefix, 'lib')
      self.binDir     = os.path.join(self.prefix, 'bin')
    if not os.path.isdir(self.includeDir):
      os.mkdir(self.includeDir)
    if not os.path.isdir(self.libDir):
      os.mkdir(self.libDir)
    if not os.path.isdir(self.binDir):
      os.mkdir(self.binDir)
    return

  def buildShowme(self, builder):
    '''Builds the ShowMe mesh viewer'''
    builder.pushConfiguration('ShowMe Binary')
    builder.setCompilerFlags(self.X11.include)
    builder.setLinkerExtraArguments(self.X11.lib)
    #builder.getLinkerObject().libraries.add(self.X11.lib)
    source = os.path.join('src', 'showme.c')
    self.builder.compile([source])
    self.builder.link([self.builder.getCompilerTarget(source)], os.path.join(self.binDir, 'showme'))
    builder.popConfiguration()
    return

  def buildTriangle(self, builder):
    '''Builds the Triangle mesh generator'''
    builder.pushConfiguration('Triangle Binary')
    builder.setCompilerFlags(' '.join(['-DNO_PETSC_MALLOC']))
    source = os.path.join('src', 'triangle.c')
    if self.libraries.math:
      builder.getLinkerObject().libraries.update(self.libraries.math)
    builder.compile([source])
    builder.link([self.builder.getCompilerTarget(source)], os.path.join(self.binDir, 'triangle'))
    builder.popConfiguration()
    return

  def buildLibrary(self, builder):
    '''Builds the Triangle mesh generation library'''
    builder.pushConfiguration('Triangle Library')
    builder.setCompilerFlags(' '.join(['-DNO_PETSC_MALLOC', '-DTRILIBRARY']))
    builder.getCompilerObject().includeDirectories.add('src')
    source = os.path.join('src', 'triangle.c')
    object = os.path.join('src', 'libtriangle.o')
    if self.libraries.math:
      builder.getSharedLinkerObject().libraries.update(self.libraries.math)
    builder.compile([source], object)
    builder.link([object], os.path.join(self.libDir, 'libtriangle.'+self.setCompilers.sharedLibraryExt), shared = 1)
    builder.popConfiguration()
    return

  def buildIncludes(self, builder):
    import shutil
    if os.path.isfile(os.path.join(self.includeDir, 'triangle.h')):
      os.remove(os.path.join(self.includeDir, 'triangle.h'))
    if os.path.isfile(os.path.join(self.includeDir, 'config.h')):
      os.remove(os.path.join(self.includeDir, 'config.h'))
    shutil.copy(os.path.join('src', 'triangle.h'), os.path.join(self.includeDir, 'triangle.h'))
    shutil.copy(os.path.join('src', 'config.h'), os.path.join(self.includeDir, 'config.h'))
    return

  def updateDependencies(self, sourceDB):
    sourceDB.updateSource(os.path.join('src', 'triangle.h'))
    maker.Make.updateDependencies(self, sourceDB)
    return

  def build(self, builder, setupOnly):
    if setupOnly:
      return
    self.buildDirectories(builder)
    self.buildShowme(builder)
    self.buildTriangle(builder)
    self.buildLibrary(builder)
    self.buildIncludes(builder)
    return

if __name__ == '__main__':
  Make().run()
