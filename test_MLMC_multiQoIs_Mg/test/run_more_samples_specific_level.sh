#!/bin/bash

for i in $(seq 10); do
	python3 wrapper_multilevel_multiple_qoi.py --level=4 --nb_of_qoi=10
done

