#! /bin/bash

cal=1

[ ! -d "./output" ] && mkdir output

echo "#! /bin/bash" > output/00combinePDF.sh
echo -n "pdftk " >> output/00combinePDF.sh

for i in {1..4} # 10
do
    echo "case: $i"
    export case=$i
    if [ ${cal} == "1" ] ; then
	nohup julia simu.jl > output/simu${i}.out &
    fi
    echo -n "0case${i}.pdf " >> output/00combinePDF.sh
    sleep 0.1s
done

echo "output 00mse.pdf" >> output/00combinePDF.sh
echo "sleep 0.1s" >> output/00combinePDF.sh
chmod +x output/00combinePDF.sh
