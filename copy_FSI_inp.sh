#!/usr/bin/bash
for i in {002..009}; do 
    echo $i
    cp FSI_case_001.inp FSI_case_$i.inp
    sed -i 1s/001/$i/ FSI_case_$i.inp
done

