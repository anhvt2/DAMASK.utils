#! /bin/bash

FINISHED=0

while [[ $FINISHED < 1 ]]
do
    OUTPUT=$(julia Example.jl 2>&1 | tee /dev/tty)
	if ! echo $OUTPUT | grep -q "Written parameter values"
    then
        FINISHED=1
    else
        python3 take_samples.py 
    fi
done
