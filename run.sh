#!/bin/bash

while  read line
do 
    python code/train.py $line

done < experiments.txt