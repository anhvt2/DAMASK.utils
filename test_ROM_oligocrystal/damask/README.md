
This folder controls the DAMASK template and its successful descendants
	* with various loading rate in `tension.load`
	* various temperature by adding `{./initialT.config}` in `material.config`
	* create a series of folders, e.g. `1/`, `2/`, ..., `1000/` with various strain rates and initial temperature
	* create a backup/log for loading rate and temperature in `control.log`

See `randomizeLoad.py` for more details
