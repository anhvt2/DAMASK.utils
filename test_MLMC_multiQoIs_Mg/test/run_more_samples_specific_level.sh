#!/bin/bash

for i in $(seq 200); do
	python3 wrapper_multilevel_multiple_qoi.py --level=2 --nb_of_qoi=10
done

