# Copyright 2011-2022 Max-Planck-Institut für Eisenforschung GmbH
# 
# DAMASK is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import numpy as np
import h5py
from typing import Sequence, Dict, Any, Collection

from ._typehints import FileHandle
from . import Config
from . import Rotation
from . import Orientation
from . import util
from . import Table


class ConfigMaterial(Config):
    """
    Material configuration.

    Manipulate material configurations for storage in YAML format.
    A complete material configuration file has the entries 'material',
    'phase', and 'homogenization'. For use in DAMASK, it needs to be
    stored as 'material.yaml'.

    """

    def __init__(self,
                 d: Dict[str, Any] = None,
                 **kwargs):
        """
        New material configuration.

        Parameters
        ----------
        d : dictionary or YAML string, optional
            Initial content. Defaults to None, in which case empty entries for
            any missing material, homogenization, and phase entry are created.
        kwargs : key=value pairs, optional
            Initial content specified as pairs of key=value.

        """
        default: Collection
        if d is None:
            for section, default in {'material':[],'homogenization':{},'phase':{}}.items():
                if section not in kwargs: kwargs.update({section:default})

        super().__init__(d,**kwargs)


    def save(self,
             fname: FileHandle = 'material.yaml',
             **kwargs):
        """
        Save to yaml file.

        Parameters
        ----------
        fname : file, str, or pathlib.Path, optional
            Filename or file for writing. Defaults to 'material.yaml'.
        **kwargs
            Keyword arguments parsed to yaml.dump.

        """
        super().save(fname,**kwargs)


    @classmethod
    def load(cls,
             fname: FileHandle = 'material.yaml') -> 'ConfigMaterial':
        """
        Load from yaml file.

        Parameters
        ----------
        fname : file, str, or pathlib.Path, optional
            Filename or file to read from. Defaults to 'material.yaml'.

        Returns
        -------
        loaded : damask.ConfigMaterial
            Material configuration from file.

        """
        return super(ConfigMaterial,cls).load(fname)


    @staticmethod
    def load_DREAM3D(fname: str,
                     grain_data: str = None,
                     cell_data: str = None,
                     cell_ensemble_data: str = 'CellEnsembleData',
                     phases: str = 'Phases',
                     Euler_angles: str = 'EulerAngles',
                     phase_names: str = 'PhaseName',
                     base_group: str = None) -> 'ConfigMaterial':
        """
        Load DREAM.3D (HDF5) file.

        Data in DREAM.3D files can be stored per cell ('CellData')
        and/or per grain ('Grain Data'). Per default, cell-wise data
        is assumed.

        damask.Grid.load_DREAM3D allows to get the corresponding geometry
        for the grid solver.

        Parameters
        ----------
        fname : str
            Filename of the DREAM.3D (HDF5) file.
        grain_data : str
            Name of the group (folder) containing grain-wise data. Defaults
            to None, in which case cell-wise data is used.
        cell_data : str
            Name of the group (folder) containing cell-wise data. Defaults to
            None in wich case it is automatically detected.
        cell_ensemble_data : str
            Name of the group (folder) containing data of cell ensembles. This
            group is used to inquire the name of the phases. Phases will get
            numeric IDs if this group is not found. Defaults to 'CellEnsembleData'.
        phases : str
            Name of the dataset containing the phase ID (cell-wise or grain-wise).
            Defaults to 'Phases'.
        Euler_angles : str
            Name of the dataset containing the crystallographic orientation as
            Euler angles in radians (cell-wise or grain-wise). Defaults to 'EulerAngles'.
        phase_names : str
            Name of the dataset containing the phase names. Phases will get
            numeric IDs if this dataset is not found. Defaults to 'PhaseName'.
        base_group : str
            Path to the group (folder) that contains geometry (_SIMPL_GEOMETRY),
            and grain- or cell-wise data. Defaults to None, in which case
            it is set as the path that contains _SIMPL_GEOMETRY/SPACING.

        Notes
        -----
        Homogenization and phase entries are emtpy and need to be defined separately.

        Returns
        -------
        loaded : damask.ConfigMaterial
            Material configuration from file.

        """
        b = util.DREAM3D_base_group(fname) if base_group is None else base_group
        c = util.DREAM3D_cell_data_group(fname) if cell_data is None else cell_data
        f = h5py.File(fname,'r')

        if grain_data is None:
            phase = f['/'.join([b,c,phases])][()].flatten()
            O = Rotation.from_Euler_angles(f['/'.join([b,c,Euler_angles])]).as_quaternion().reshape(-1,4) # noqa
            _,idx = np.unique(np.hstack([O,phase.reshape(-1,1)]),return_index=True,axis=0)
            idx = np.sort(idx)
        else:
            phase = f['/'.join([b,grain_data,phases])][()]
            O = Rotation.from_Euler_angles(f['/'.join([b,grain_data,Euler_angles])]).as_quaternion() # noqa
            idx = np.arange(phase.size)

        if cell_ensemble_data is not None and phase_names is not None:
            try:
                names = np.array([s.decode() for s in f['/'.join([b,cell_ensemble_data,phase_names])]])
                phase = names[phase]
            except KeyError:
                pass


        base_config = ConfigMaterial({'phase':{k if isinstance(k,int) else str(k):'t.b.d.' for k in np.unique(phase)},
                                      'homogenization':{'direct':{'N_constituents':1}}})
        constituent = {k:np.atleast_1d(v[idx].squeeze()) for k,v in zip(['O','phase'],[O,phase])}

        return base_config.material_add(**constituent,homogenization='direct')


    @staticmethod
    def from_table(table: Table,
                   **kwargs) -> 'ConfigMaterial':
        """
        Generate from an ASCII table.

        Parameters
        ----------
        table : damask.Table
            Table that contains material information.
        **kwargs
            Keyword arguments where the key is the property name and
            the value specifies either the label of the data column in the table
            or a constant value.

        Returns
        -------
        new : damask.ConfigMaterial
            Material configuration from values in table.

        Examples
        --------
        >>> import damask
        >>> import damask.ConfigMaterial as cm
        >>> t = damask.Table.load('small.txt')
        >>> t
            pos  pos  pos   qu   qu    qu    qu   phase    homog
        0    0    0    0  0.19  0.8   0.24 -0.51  Aluminum SX
        1    1    0    0  0.8   0.19  0.24 -0.51  Steel    SX
        1    1    1    0  0.8   0.19  0.24 -0.51  Steel    SX
        >>> cm.from_table(t,O='qu',phase='phase',homogenization='homog')
        material:
          - constituents:
              - O: [0.19, 0.8, 0.24, -0.51]
                v: 1.0
                phase: Aluminum
            homogenization: SX
          - constituents:
              - O: [0.8, 0.19, 0.24, -0.51]
                v: 1.0
                phase: Steel
            homogenization: SX
        homogenization: {}
        phase: {}

        >>> cm.from_table(t,O='qu',phase='phase',homogenization='single_crystal')
        material:
          - constituents:
              - O: [0.19, 0.8, 0.24, -0.51]
                v: 1.0
                phase: Aluminum
            homogenization: single_crystal
          - constituents:
              - O: [0.8, 0.19, 0.24, -0.51]
                v: 1.0
                phase: Steel
            homogenization: single_crystal
        homogenization: {}
        phase: {}

        """
        kwargs_ = {k:table.get(v) if v in table.labels else np.atleast_2d([v]*len(table)).T for k,v in kwargs.items()}

        _,idx = np.unique(np.hstack(list(kwargs_.values())),return_index=True,axis=0)
        idx = np.sort(idx)
        kwargs_ = {k:np.atleast_1d(v[idx].squeeze()) for k,v in kwargs_.items()}

        return ConfigMaterial().material_add(**kwargs_)


    @property
    def is_complete(self) -> bool:
        """
        Check for completeness.

        Only the general file layout is considered.
        This check does not consider whether parameters for
        a particular phase/homogenization model are missing.

        Returns
        -------
        complete : bool
            Whether the material.yaml definition is complete.

        """
        ok = True
        for top_level in ['homogenization','phase','material']:
            ok &= top_level in self
            if top_level not in self: print(f'{top_level} entry missing')

        if ok:
           ok &= len(self['material']) > 0
           if len(self['material']) < 1: print('Incomplete material definition')

        if ok:
            homogenization = set()
            phase          = set()
            for i,v in enumerate(self['material']):
                if 'homogenization' in v:
                    homogenization.add(v['homogenization'])
                else:
                    print(f'No homogenization specified in material {i}')
                    ok = False

                if 'constituents' in v:
                    for ii,vv in enumerate(v['constituents']):
                        if 'O' not in vv:
                            print('No orientation specified in constituent {ii} of material {i}')
                            ok = False
                        if 'phase' in vv:
                            phase.add(vv['phase'])
                        else:
                            print(f'No phase specified in constituent {ii} of material {i}')
                            ok = False

            for k,v in self['phase'].items():
                if 'lattice' not in v:
                    print(f'No lattice specified in phase {k}')
                    ok = False

            for k,v in self['homogenization'].items():
                if 'N_constituents' not in v:
                    print(f'No. of constituents not specified in homogenization {k}')
                    ok = False

            if phase - set(self['phase']):
                print(f'Phase(s) {phase-set(self["phase"])} missing')
                ok = False
            if homogenization - set(self['homogenization']):
                print(f'Homogenization(s) {homogenization-set(self["homogenization"])} missing')
                ok = False
        return ok


    @property
    def is_valid(self) -> bool:
        """
        Check for valid content.

        Only the generic file content is considered.
        This check does not consider whether parameters for a
        particular phase/homogenization mode are out of bounds.

        Returns
        -------
        valid : bool
            Whether the material.yaml definition is valid.

        """
        ok = True

        if 'phase' in self:
            for k,v in self['phase'].items():
                if 'lattice' in v:
                    try:
                        Orientation(lattice=v['lattice'])
                    except KeyError:
                        print(f"Invalid lattice '{v['lattice']}' in phase '{k}'")
                        ok = False

        if 'material' in self:
            for i,m in enumerate(self['material']):
                if 'constituents' in m:
                    v = 0.0
                    for c in m['constituents']:
                        v += float(c['v'])
                        if 'O' in c:
                            try:
                                Rotation.from_quaternion(c['O'])
                            except ValueError:
                                print(f"Invalid orientation '{c['O']}' in material '{i}'")
                                ok = False
                    if not np.isclose(v,1.0):
                        print(f"Total fraction v = {v} ≠ 1 in material '{i}'")
                        ok = False

        return ok


    def material_rename_phase(self,
                              mapping: Dict[str, str],
                              ID: Sequence[int] = None,
                              constituent: Sequence[int] = None) -> 'ConfigMaterial':
        """
        Change phase name in material.

        Parameters
        ----------
        mapping: dictionary
            Mapping from old name to new name
        ID: list of ints, optional
            Limit renaming to selected material IDs.
        constituent: list of ints, optional
            Limit renaming to selected constituents.

        Returns
        -------
        updated : damask.ConfigMaterial
            Updated material configuration.

        """
        dup = self.copy()
        for i,m in enumerate(dup['material']):
            if ID is not None and i not in ID: continue
            for c in m['constituents']:
                if constituent is not None and c not in constituent: continue
                try:
                    c['phase'] = mapping[c['phase']]
                except KeyError:
                    continue
        return dup


    def material_rename_homogenization(self,
                                       mapping: Dict[str, str],
                                       ID: Sequence[int] = None) -> 'ConfigMaterial':
        """
        Change homogenization name in material.

        Parameters
        ----------
        mapping: dictionary
            Mapping from old name to new name
        ID: list of ints, optional
            Limit renaming to selected homogenization IDs.

        Returns
        -------
        updated : damask.ConfigMaterial
            Updated material configuration.

        """
        dup = self.copy()
        for i,m in enumerate(dup['material']):
            if ID is not None and i not in ID: continue
            try:
                m['homogenization'] = mapping[m['homogenization']]
            except KeyError:
                continue
        return dup


    def material_add(self,
                     **kwargs: Any) -> 'ConfigMaterial':
        """
        Add material entries.

        Parameters
        ----------
        **kwargs
            Key-value pairs.

        Returns
        -------
        updated : damask.ConfigMaterial
            Updated material configuration.

        Examples
        --------
        Create a dual-phase steel microstructure for micromechanical simulations:

        >>> import numpy as np
        >>> import damask
        >>> m = damask.ConfigMaterial()
        >>> m = m.material_add(phase = ['Ferrite','Martensite'],
        ...                    O = damask.Rotation.from_random(2),
        ...                    homogenization = 'SX')
        >>> m
        material:
          - constituents:
              - O: [0.577764, -0.146299, -0.617669, 0.513010]
                v: 1.0
                phase: Ferrite
            homogenization: SX
          - constituents:
              - O: [0.184176, 0.340305, 0.737247, 0.553840]
                v: 1.0
                phase: Martensite
            homogenization: SX
        homogenization: {}
        phase: {}

        Create a duplex stainless steel microstructure for forming simulations:

        >>> import numpy as np
        >>> import damask
        >>> m = damask.ConfigMaterial()
        >>> m = m.material_add(phase = np.array(['Austenite','Ferrite']).reshape(1,2),
        ...                    O = damask.Rotation.from_random((2,2)),
        ...                    v = np.array([0.2,0.8]).reshape(1,2),
        ...                    homogenization = 'Taylor')
        >>> m
        material:
          - constituents:
              - phase: Austenite
                O: [0.659802978293224, 0.6953785848195171, 0.22426295326327111, -0.17554139512785227]
                v: 0.2
              - phase: Ferrite
                O: [0.49356745891301596, 0.2841806579193434, -0.7487679215072818, -0.339085707289975]
                v: 0.8
            homogenization: Taylor
          - constituents:
              - phase: Austenite
                O: [0.26542221365204055, 0.7268854930702071, 0.4474726435701472, -0.44828201137283735]
                v: 0.2
              - phase: Ferrite
                O: [0.6545817158479885, -0.08004812803625233, -0.6226561293931374, 0.4212059104577611]
                v: 0.8
            homogenization: Taylor
        homogenization: {}
        phase: {}

        """
        N,n,shaped = 1,1,{}

        map_dim = {'O':-1,'V_e':-2}
        for k,v in kwargs.items():
            shaped[k] = np.array(v)
            s = shaped[k].shape[:map_dim.get(k,None)]
            N = max(N,s[0]) if len(s)>0 else N
            n = max(n,s[1]) if len(s)>1 else n

        mat: Sequence[dict] = [{'constituents':[{} for _ in range(n)]} for _ in range(N)]

        if 'v' not in kwargs:
            shaped['v'] = np.broadcast_to(1/n,(N,n))

        map_shape = {'O':(N,n,4),'V_e':(N,n,3,3)}
        for k,v in shaped.items():
            target = map_shape.get(k,(N,n))
            obj = np.broadcast_to(v.reshape(util.shapeshifter(v.shape, target, mode = 'right')), target)
            for i in range(N):
                if k in ['phase','O','v','V_e']:
                    for j in range(n):
                        mat[i]['constituents'][j][k] = obj[i,j].item() if isinstance(obj[i,j],np.generic) else obj[i,j]
                else:
                    mat[i][k] = obj[i,0].item() if isinstance(obj[i,0],np.generic) else obj[i,0]

        dup = self.copy()
        dup['material'] = dup['material'] + mat if 'material' in dup else mat

        return dup
