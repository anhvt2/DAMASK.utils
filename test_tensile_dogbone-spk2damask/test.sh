#!/bin/bash
python3 geom_cad2phase.py -r 50 -d 'dump.12.out' -L 10000 -W 6000 -T 1000 -l 4000 -w 1000 -b 1000 -R 10
python3 geom_spk2dmsk.py -r 50 -d 'dump.12.out'
geom_check spk_dump_12_out.geom

