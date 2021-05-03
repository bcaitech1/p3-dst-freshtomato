#!/bin/bash

while  read line
do 
    python3 code/train.py $line

done < experiments.txt