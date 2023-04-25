#!/bin/bash

# grep -inr 'at 4' log.MultilevelEstimators-multiQoIs.2 > tmp.txt
# then extract the first column
for i in $(cat tmp.txt); do 
	sed -n ${i}p ../log.MultilevelEstimators-multiQoIs.2
	j=$((i+1))
	sed -n ${j}p ../log.MultilevelEstimators-multiQoIs.2
done
