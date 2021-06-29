#!/usr/bin/env python3
# Copyright 2011-20 Max-Planck-Institut für Eisenforschung GmbH
# 
# DAMASK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import os
import glob
import argparse
from pathlib import Path

import damask

msc_version = float(damask.environment.options['MSC_VERSION'])
if int(msc_version) == msc_version:
    msc_version = int(msc_version)
msc_root     = Path(damask.environment.options['MSC_ROOT'])
damask_root  = damask.environment.root_dir

parser = argparse.ArgumentParser(
     description='Apply DAMASK modification to MSC.Marc/Mentat',
     epilog = f'MSC_ROOT={msc_root} and MSC_VERSION={msc_version} (from {damask_root}/env/CONFIG)')
parser.add_argument('--editor', dest='editor', metavar='string', default='vi',
                    help='Name of the editor for MSC.Mentat (executable)')


def copy_and_replace(in_file,dst):
    with open(in_file) as f:
        content = f.read()
    content = content.replace('%INSTALLDIR%',str(msc_root))
    content = content.replace('%VERSION%',str(msc_version))
    content = content.replace('%EDITOR%', parser.parse_args().editor)
    with open(dst/Path(in_file).name,'w') as f:
        f.write(content)


print('adapting Marc tools...\n')

src = damask_root/f'installation/mods_MarcMentat/{msc_version}/Marc_tools'
dst = msc_root/f'marc{msc_version}/tools'
for in_file in glob.glob(str(src/'*damask*')) + [str(src/'include_linux64')]:
    copy_and_replace(in_file,dst)


print('adapting Mentat scripts and menus...\n')

src = damask_root/f'installation/mods_MarcMentat/{msc_version}/Mentat_bin'
dst = msc_root/f'mentat{msc_version}/bin'
for in_file in glob.glob(str(src/'*[!.original]')):
    copy_and_replace(in_file,dst)

src = damask_root/f'installation/mods_MarcMentat/{msc_version}/Mentat_menus'
dst = msc_root/f'mentat{msc_version}/menus'
for in_file in glob.glob(str(src/'job_run.ms')):
    copy_and_replace(in_file,dst)


print('compiling Mentat menu binaries...')

executable = str(msc_root/f'mentat{msc_version}/bin/mentat')
menu_file  = str(msc_root/f'mentat{msc_version}/menus/linux64/main.msb')
os.system(f'xvfb-run {executable} -compile {menu_file}')


print('setting file access rights...\n')

for pattern in [msc_root/f'marc{msc_version}/tools/*damask*',
                msc_root/f'mentat{msc_version}/bin/submit?',
                msc_root/f'mentat{msc_version}/bin/kill?']:
    for f in glob.glob(str(pattern)):
        os.chmod(f,0o755)
