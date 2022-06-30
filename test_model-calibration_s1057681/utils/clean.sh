#!/bin/bash
for i in $(seq 4 5000); do
	rm -rfv fpga_Iter${i}
done
