#! /bin/bash

for i in {1..4} # 10
do
    echo "case: $i"
    export case=$i
    nohup julia full-simu.jl > output/full-simu${i}.out &
    sleep 0.1s
done
