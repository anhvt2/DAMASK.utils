#!/bin/bash
# python3 geom_cad2phase.py -r 50 -d 'dump.12.out' -L 10000 -W 6000 -T 1000 -l 4000 -w 1000 -b 1000 -R 1000
# python3 geom_spk2dmsk.py -r 50 -d 'dump.12.out'
# geom_check spk_dump_12_out.geom

# python3 geom_cad2phase.py -r 10 -d 'dump.10.out' -L 10000 -W 6000 -T 1000 -l 4000 -w 1000 -b 1000 -R 1000
# python3 geom_spk2dmsk.py  -r 10 -d 'dump.10.out'
# geom_check spk_dump_10_out.geom

python3 geom_cad2phase.py -r 20 -d 'dump.20.out' -L 10000 -W 6000 -T 1000 -l 4000 -w 1000 -b 1000 -R 1000
python3 geom_spk2dmsk.py  -r 20 -d 'dump.20.out'
geom_check spk_dump_20_out.geom

cd damask/
cat material.config.preamble  | cat - material.config | sponge material.config
