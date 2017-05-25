#!/bin/bash

chrlist=(14 15 16 17 18 19 20 21 22 X Y)
chrindex=0
for sid in 01 02 03 04 05 06 07 08 09 10 11; do
  ssh pangaofeng@slave$sid "
    cd ~/share/label; 
    for i in \`seq 0 22\`; do 
      ./parse-chr-peak.sh chr${chrlist[$chrindex]} \$i 40 &> /dev/null < /dev/null & 
      #echo chr${chrlist[$chrindex]}
    done 
    disown -a
  "
  chrindex=$(( chrindex + 1 ))
done
